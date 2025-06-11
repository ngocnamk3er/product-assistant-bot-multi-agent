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
        "id": "event101",
        "text": "Sự kiện ra mắt dòng điện thoại thông minh Nova Z10 sẽ được tổ chức vào ngày 20 tháng 9 tại TP. Hồ Chí Minh. Nova Z10 nổi bật với camera AI 200MP và thiết kế viền mỏng.",
        "source": "Thông cáo báo chí từ Nova Mobile",
    },
    {
        "id": "event102",
        "text": "Đặt trước Nova Z10 từ ngày 21 đến 30 tháng 9 để nhận ngay ưu đãi giảm giá 15% và tai nghe không dây Nova Buds Pro trị giá 2 triệu đồng.",
        "source": "Trang chủ Nova Mobile - Chương trình đặt trước",
    },
    {
        "id": "event103",
        "text": "Hãng điện thoại VinaTech công bố xây dựng nhà máy sản xuất smartphone thế hệ mới tại Bắc Ninh, dự kiến đi vào hoạt động đầu năm 2026.",
        "source": "Báo Công Nghệ Việt - Tin tức sản xuất",
    },
    {
        "id": "event104",
        "text": "Chiến dịch 'Flash Sale 48h' giảm giá tới 40% cho các dòng điện thoại Galaxy Edge và Galaxy Lite, diễn ra từ 10 đến 12 tháng 10.",
        "source": "Trang khuyến mãi chính thức của Galaxy Việt Nam",
    },
    {
        "id": "event105",
        "text": "TechVision hợp tác với AI Quantum Labs để phát triển chip xử lý mới dành cho smartphone, tăng hiệu suất gấp 3 lần so với thế hệ trước.",
        "source": "Thông cáo hợp tác công nghệ TechVision 2024",
    },
    {
        "id": "event106",
        "text": "Cuộc thi ảnh 'Shot on Pixel Z' dành riêng cho người dùng điện thoại Pixel Z vừa được khởi động, với tổng giải thưởng hơn 100 triệu đồng.",
        "source": "Website Pixel Việt Nam - Thông báo cuộc thi",
    },
    {
        "id": "event107",
        "text": "Điện thoại Phoenix S9 đạt chứng nhận kháng nước IP69 và được trang bị kính chống vỡ Corning Z-Shield mới nhất.",
        "source": "Bản tin sản phẩm Phoenix S9 - Tạp chí Công nghệ",
    },
    {
        "id": "event108",
        "text": "Từ ngày 1 đến 7 tháng 11, khi mua bất kỳ mẫu điện thoại nào của thương hiệu Xphone tại cửa hàng chính hãng, khách hàng sẽ được giảm giá 1 triệu đồng và nhận thêm sạc nhanh 65W miễn phí.",
        "source": "Fanpage Xphone Việt Nam",
    },
    {
        "id": "event109",
        "text": "Hội nghị Nhà phát triển Ứng dụng di động sẽ có phiên chuyên đề giới thiệu về các tính năng độc quyền trên điện thoại Galaxy Note V.",
        "source": "Lịch trình Hội nghị Mobile DevCon 2024",
    },
    {
        "id": "event110",
        "text": "Cửa hàng Thế Giới Số khai trương chi nhánh mới tại Đà Nẵng, mở bán 500 chiếc điện thoại OneMax 12 với giá ưu đãi chỉ 3.990.000đ.",
        "source": "Thông báo khai trương từ Thế Giới Số",
    },
    {
        "id": "event111",
        "text": "SkyPhone ra mắt công nghệ 'Pin kép phân tách nhiệt' giúp tăng tuổi thọ pin thêm 35% so với công nghệ hiện tại.",
        "source": "Bản tin Công nghệ SkyPhone 2024",
    },
    {
        "id": "event112",
        "text": "Sự kiện 'Trải nghiệm trước - Mua sau' dành riêng cho dòng smartphone Aven X Series tổ chức tại Hà Nội và TP. Hồ Chí Minh cuối tuần này.",
        "source": "Thông báo sự kiện từ Aven Việt Nam",
    },
    {
        "id": "event113",
        "text": "Mẫu điện thoại Vega 8 Pro giành giải 'Thiết kế đột phá' tại Triển lãm Công nghệ Châu Á 2024 nhờ mặt lưng biến màu theo ánh sáng.",
        "source": "Báo cáo giải thưởng TechAsia 2024",
    },
    {
        "id": "event114",
        "text": "Vivo tổ chức livestream ra mắt dòng sản phẩm mới Vivo Z90 kèm mini game trúng điện thoại và voucher trị giá 500.000đ.",
        "source": "Fanpage chính thức Vivo Việt Nam",
    },
    {
        "id": "event115",
        "text": "FPT Shop công bố chương trình đổi cũ lấy mới: thu mua điện thoại cũ lên tới 3 triệu đồng khi nâng cấp lên mẫu Galaxy Z Fold mới nhất.",
        "source": "Website FPT Shop - Trang chương trình thu cũ đổi mới",
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
