import os
from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# from google.generativeai import embed_content # Nếu dùng trực tiếp
# from google import generativeai as genai      # Hoặc cách này
from google import genai  # Sử dụng cách import giống agent của bạn
from google.genai.types import EmbedContentConfig

load_dotenv()

# -----------------------------------------------------------------------------
# ⚙️ Configuration (MATCH THIS WITH YOUR AGENT'S CONFIG)
# -----------------------------------------------------------------------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables.")
# genai.configure(api_key=GOOGLE_API_KEY)


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = "event_knowledge_base"  # <<< PHẢI KHỚP VỚI AGENT
EMBEDDING_MODEL_NAME = "embedding-001"
# Biết dimension của model embedding là rất quan trọng!
# embedding-001 thường có 768 dimensions. Kiểm tra tài liệu của Google.``
EMBEDDING_DIMENSION = 768  # <<< THAY ĐỔI NẾU MODEL CỦA BẠN KHÁC

# Tên các trường trong Milvus (PHẢI KHỚP VỚI AGENT)
ID_FIELD_NAME = "doc_id"  # Thêm trường ID để dễ quản lý
VECTOR_FIELD_NAME = "embedding"
TEXT_CONTENT_FIELD_NAME = "text_content"
SOURCE_FIELD_NAME = "source_document"  # Tùy chọn

# -----------------------------------------------------------------------------
#  SAMPLE DATA TO SEED
# -----------------------------------------------------------------------------
SAMPLE_DATA = [
    {
        "id": "event001",  # ID duy nhất cho mỗi document
        "text": "The Annual Summer Music Festival will take place from July 15th to July 17th. Headliners include The Cosmic Keys and Solar Flare.",
        "source": "Official Festival Announcement 2024",
    },
    {
        "id": "event002",
        "text": "Early bird tickets for the Summer Music Festival are available until June 30th. VIP passes include backstage access and a dedicated lounge.",
        "source": "Festival Ticketing Page",
    },
    {
        "id": "product001",
        "text": "Our new Quantum Laptop X features a 16-core processor, 32GB RAM, and a self-healing Nanonite screen. Pre-orders open next week.",
        "source": "Product Launch Press Release - Quantum Laptop X",
    },
    {
        "id": "product002",
        "text": "The Quantum Laptop X battery lasts up to 20 hours on a single charge. It supports fast charging, reaching 80% in just 45 minutes.",
        "source": "Quantum Laptop X - Technical Specifications",
    },
    {
        "id": "event003",
        "text": "The Tech Innovators Conference is scheduled for October 5th-7th. Keynote speakers include Dr. Aris Thorne and CEO Jian Li.",
        "source": "Tech Conference Website 2024",
    },
    {
        "id": "event004",
        "text": "Call for papers for the Tech Innovators Conference is now open. Submit your research on AI, blockchain, or quantum computing by August 15th.",
        "source": "Tech Conference - Call for Papers",
    },
]


# -----------------------------------------------------------------------------
# 🛠️ Helper Function for Embedding (MATCH THIS WITH YOUR AGENT'S)
# -----------------------------------------------------------------------------
def get_text_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    """Generates embedding for the given text using Google's API."""
    try:
        result = genai.Client().models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=text,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",  # Optional
                output_dimensionality=768,  # Optional
                title="Driver's License",  # Optional
            ),
        )
        # print(f"Generated embedding for text: {result.embeddings[0].values}")  # Debugging
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        return []


# -----------------------------------------------------------------------------
# 🚀 Main Seeding Logic
# -----------------------------------------------------------------------------
def seed_milvus_data():
    # 1. Connect to Milvus
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    # 2. Define Schema and Create Collection (if it doesn't exist)
    # Primary key field
    id_field = FieldSchema(
        name=ID_FIELD_NAME,
        dtype=DataType.VARCHAR,  # Hoặc INT64 nếu ID của bạn là số
        is_primary=True,
        auto_id=False,  # Chúng ta sẽ tự cung cấp ID từ SAMPLE_DATA
        max_length=100,  # Điều chỉnh nếu ID của bạn dài hơn
    )
    # Vector embedding field
    embedding_field = FieldSchema(
        name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION
    )
    # Text content field
    text_content_field = FieldSchema(
        name=TEXT_CONTENT_FIELD_NAME,
        dtype=DataType.VARCHAR,
        max_length=65535,  # Giới hạn tối đa của VARCHAR trong Milvus
    )
    # Source document field (optional)
    source_field = FieldSchema(
        name=SOURCE_FIELD_NAME, dtype=DataType.VARCHAR, max_length=1024
    )

    schema_fields = [id_field, embedding_field, text_content_field]
    if SOURCE_FIELD_NAME:  # Chỉ thêm nếu được định nghĩa
        schema_fields.append(source_field)

    schema = CollectionSchema(
        fields=schema_fields,
        description="Knowledge base for events and products",
        enable_dynamic_field=False,  # Đặt True nếu bạn muốn thêm các trường không xác định trước
    )

    if utility.has_collection(MILVUS_COLLECTION_NAME):
        print(
            f"Collection '{MILVUS_COLLECTION_NAME}' already exists. Skipping creation."
        )
        # Cân nhắc: Bạn có muốn xóa và tạo lại collection không?
        # utility.drop_collection(MILVUS_COLLECTION_NAME)
        # print(f"Dropped existing collection '{MILVUS_COLLECTION_NAME}'.")
        # collection = Collection(MILVUS_COLLECTION_NAME, schema=schema)
        # print(f"Re-created collection '{MILVUS_COLLECTION_NAME}'.")
        collection = Collection(MILVUS_COLLECTION_NAME)
    else:
        print(f"Creating collection '{MILVUS_COLLECTION_NAME}'...")
        collection = Collection(MILVUS_COLLECTION_NAME, schema=schema)
        print(f"Collection '{MILVUS_COLLECTION_NAME}' created successfully.")

    # 3. Prepare data for insertion
    print("\nPreparing data and generating embeddings...")
    data_to_insert = []  # List of lists, or list of dicts

    # Chuẩn bị dữ liệu theo định dạng list của các list, mỗi list con tương ứng với 1 entity
    # theo thứ tự các field trong schema (trừ primary key nếu auto_id=True)
    # Hoặc có thể dùng list của các dictionary nếu phiên bản PyMilvus hỗ trợ (thường là mới hơn)

    ids_to_insert = []
    embeddings_to_insert = []
    texts_to_insert = []
    sources_to_insert = []  # Chỉ khi SOURCE_FIELD_NAME được dùng

    for item in SAMPLE_DATA:
        text_to_embed = item["text"]
        embedding = get_text_embedding(text_to_embed)
        if embedding:  # Chỉ thêm nếu embedding thành công
            ids_to_insert.append(item["id"])
            embeddings_to_insert.append(embedding)
            texts_to_insert.append(item["text"])
            if SOURCE_FIELD_NAME:
                sources_to_insert.append(item.get("source", "N/A"))
            print(f"  Generated embedding for ID: {item['id']}")
        else:
            print(f"  Skipping ID: {item['id']} due to embedding failure.")

    if not ids_to_insert:
        print("No data to insert after embedding process. Exiting.")
        return

    # Xây dựng list of lists cho việc insert
    # Thứ tự phải khớp với schema_fields (trừ primary key nếu auto_id=True, nhưng ở đây auto_id=False)
    if SOURCE_FIELD_NAME:
        entities_to_insert = [
            ids_to_insert,
            embeddings_to_insert,
            texts_to_insert,
            sources_to_insert,
        ]
    else:
        entities_to_insert = [ids_to_insert, embeddings_to_insert, texts_to_insert]

    # 4. Insert data
    if entities_to_insert[0]:  # Kiểm tra xem có dữ liệu để insert không
        print(
            f"\nInserting {len(entities_to_insert[0])} entities into '{MILVUS_COLLECTION_NAME}'..."
        )
        try:
            insert_result = collection.insert(entities_to_insert)
            print("Data inserted successfully.")
            print(f"  Primary keys of inserted entities: {insert_result.primary_keys}")
            print(f"  Number of entities inserted: {insert_result.insert_count}")

            # 5. Flush data (important to make inserts searchable)
            print("Flushing collection...")
            collection.flush()
            print("Collection flushed.")

        except Exception as e:
            print(f"Error inserting data: {e}")
            import traceback

            traceback.print_exc()
            return
    else:
        print("No valid data to insert.")
        return

    # 6. Create Index (CRUCIAL for search performance)
    # Kiểm tra xem index đã tồn tại chưa
    has_index = False
    for index_info in collection.indexes:
        if index_info.field_name == VECTOR_FIELD_NAME:
            has_index = True
            print(
                f"\nIndex on field '{VECTOR_FIELD_NAME}' already exists: {index_info.index_name}"
            )
            break

    if not has_index:
        print(f"\nCreating index for field '{VECTOR_FIELD_NAME}'...")
        # Chọn loại index và tham số phù hợp. Ví dụ: IVF_FLAT hoặc HNSW
        # IVF_FLAT:
        index_params_ivf = {
            "metric_type": "L2",  # Hoặc "IP". PHẢI KHỚP VỚI LÚC SEARCH
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 128
            },  # Số cluster, thường là sqrt(số lượng entity) đến 4*sqrt(số lượng entity)
        }
        # HNSW:
        # index_params_hnsw = {
        #     "metric_type": "L2", # Or "IP"
        #     "index_type": "HNSW",
        #     "params": {"M": 16, "efConstruction": 200} # M: max degree, efConstruction: search scope during build
        # }

        try:
            collection.create_index(
                field_name=VECTOR_FIELD_NAME,
                index_params=index_params_ivf,  # Hoặc index_params_hnsw
            )
            print(f"Index created successfully on '{VECTOR_FIELD_NAME}'.")
            utility.wait_for_index_building_complete(
                MILVUS_COLLECTION_NAME, index_name=""
            )  # Chờ index build xong
            print("Index building complete.")
        except Exception as e:
            print(f"Error creating index: {e}")
            return

    # 7. Load collection (if not already loaded for search, good for verification)
    print("\nLoading collection into memory for verification...")
    collection.load()
    print(
        f"Collection '{MILVUS_COLLECTION_NAME}' loaded. Number of entities: {collection.num_entities}"
    )

    # 8. (Optional) Compact collection (can free up space after deletions/updates, not strictly needed for new inserts)
    # print("\nCompacting collection (this might take a while for large collections)...")
    # collection.compact()
    # print("Compaction task submitted. Check Milvus logs for completion.")

    print("\nMilvus data seeding process complete.")
    connections.disconnect("default")
    print("Disconnected from Milvus.")


if __name__ == "__main__":
    seed_milvus_data()
