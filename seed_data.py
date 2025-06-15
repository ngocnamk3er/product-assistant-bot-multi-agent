# seed_data_openai.py (Phiên bản đã sửa)

import os
import openai # <<< THAY ĐỔI
from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

load_dotenv()

# -----------------------------------------------------------------------------
# ⚙️ Configuration
# -----------------------------------------------------------------------------
# --- Cấu hình OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Cấu hình Embedding (OpenAI) ---
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # <<< THAY ĐỔI: Kích thước vector của 'text-embedding-3-small'

# --- Cấu hình Milvus ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# <<< THAY ĐỔI: Đặt tên collection mới để không lẫn lộn dữ liệu
MILVUS_COLLECTION_NAME = "event_knowledge_base_openai"

# --- Tên các trường trong Milvus (giữ nguyên) ---
ID_FIELD_NAME = "doc_id"
VECTOR_FIELD_NAME = "embedding"
TEXT_CONTENT_FIELD_NAME = "text_content"
SOURCE_FIELD_NAME = "source_document"

# -----------------------------------------------------------------------------
#  SAMPLE DATA TO SEED (Không thay đổi)
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
# 🛠️ Helper Function for Embedding (Sử dụng OpenAI)
# -----------------------------------------------------------------------------
def get_text_embedding(text: str) -> list[float]:
    """
    <<< THAY ĐỔI: Tạo embedding bằng API của OpenAI.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating OpenAI embedding for text '{text[:50]}...': {e}")
        return []

# -----------------------------------------------------------------------------
# 🚀 Main Seeding Logic
# -----------------------------------------------------------------------------
def seed_milvus_data():
    # 1. Kết nối Milvus (Không thay đổi)
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    # 2. Định nghĩa Schema và Tạo Collection
    id_field = FieldSchema(
        name=ID_FIELD_NAME, dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100
    )
    # <<< THAY ĐỔI: Cập nhật dimension của vector
    embedding_field = FieldSchema(
        name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION
    )
    text_content_field = FieldSchema(
        name=TEXT_CONTENT_FIELD_NAME, dtype=DataType.VARCHAR, max_length=65535
    )
    source_field = FieldSchema(
        name=SOURCE_FIELD_NAME, dtype=DataType.VARCHAR, max_length=1024
    )

    schema = CollectionSchema(
        fields=[id_field, embedding_field, text_content_field, source_field],
        description="Knowledge base for events and products (using OpenAI embeddings)"
    )

    # Cân nhắc xóa collection cũ nếu muốn làm sạch hoàn toàn
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        print(f"Collection '{MILVUS_COLLECTION_NAME}' already exists. Dropping it to re-seed.")
        utility.drop_collection(MILVUS_COLLECTION_NAME)

    print(f"Creating collection '{MILVUS_COLLECTION_NAME}'...")
    collection = Collection(MILVUS_COLLECTION_NAME, schema=schema)
    print(f"Collection '{MILVUS_COLLECTION_NAME}' created successfully.")

    # 3. Chuẩn bị và chèn dữ liệu (Logic không đổi, chỉ gọi hàm embedding mới)
    print("\nPreparing data and generating embeddings with OpenAI...")
    ids_to_insert = []
    embeddings_to_insert = []
    texts_to_insert = []
    sources_to_insert = []

    for item in SAMPLE_DATA:
        text_to_embed = item["text"]
        embedding = get_text_embedding(text_to_embed) # Sẽ gọi hàm của OpenAI
        if embedding:
            ids_to_insert.append(item["id"])
            embeddings_to_insert.append(embedding)
            texts_to_insert.append(item["text"])
            sources_to_insert.append(item.get("source"))
            print(f"  Generated embedding for ID: {item['id']}")
        else:
            print(f"  Skipping ID: {item['id']} due to embedding failure.")

    if not ids_to_insert:
        print("No data to insert after embedding process. Exiting.")
        return

    entities_to_insert = [ids_to_insert, embeddings_to_insert, texts_to_insert, sources_to_insert]

    # 4. Chèn dữ liệu (Không thay đổi)
    if entities_to_insert[0]:
        print(f"\nInserting {len(entities_to_insert[0])} entities...")
        insert_result = collection.insert(entities_to_insert)
        print("Data inserted successfully.")
        collection.flush()
        print("Collection flushed.")
    else:
        print("No valid data to insert.")
        return

    # 5. Tạo Index (Quan trọng)
    if not collection.has_index():
        print(f"\nCreating index for field '{VECTOR_FIELD_NAME}'...")
        # <<< THAY ĐỔI: Sử dụng metric COSINE cho embedding của OpenAI
        index_params = {
            "metric_type": "COSINE",  # L2 hoặc IP cũng hoạt động, nhưng COSINE được khuyến nghị
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        try:
            collection.create_index(field_name=VECTOR_FIELD_NAME, index_params=index_params)
            print(f"Index created successfully on '{VECTOR_FIELD_NAME}'.")
            utility.wait_for_index_building_complete(MILVUS_COLLECTION_NAME)
            print("Index building complete.")
        except Exception as e:
            print(f"Error creating index: {e}")
            return
    else:
        print(f"\nIndex on field '{VECTOR_FIELD_NAME}' already exists.")

    # 6. Tải collection để xác minh (Không thay đổi)
    print("\nLoading collection into memory for verification...")
    collection.load()
    print(f"Collection '{MILVUS_COLLECTION_NAME}' loaded. Number of entities: {collection.num_entities}")
    
    print("\nMilvus data seeding process with OpenAI embeddings is complete.")
    connections.disconnect("default")
    print("Disconnected from Milvus.")


if __name__ == "__main__":
    seed_milvus_data()