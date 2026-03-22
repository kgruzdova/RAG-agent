"""
Создаёт Pinecone-индекс longtermmemory с dimension=1536 для text-embedding-3-small.
Запуск: python create_pinecone_index.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def main():
    pc_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "longtermmemory")

    if not pc_key:
        print("PINECONE_API_KEY не задан в .env")
        return 1

    print(f"Создание индекса '{index_name}' с dimension=1536...")

    try:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=pc_key)
        indexes = list(pc.list_indexes())
        existing = [idx.get("name", idx) if isinstance(idx, dict) else getattr(idx, "name", "") for idx in indexes]

        if index_name in existing:
            print(f"Индекс '{index_name}' уже существует.")
            stats = pc.Index(index_name).describe_index_stats()
            dim = getattr(stats, "dimension", None) or (stats.get("dimension") if isinstance(stats, dict) else None)
            print(f"Текущая dimension: {dim}")
            if dim != 1536:
                print("\nУдаление старого индекса и создание нового...")
                pc.delete_index(index_name)
                existing.remove(index_name)
            else:
                print("Dimension уже 1536 — всё в порядке.")
                return 0

        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Индекс '{index_name}' создан (dimension=1536).")
        print("Подожди 1–2 минуты, пока индекс станет активным, затем запусти: python check_pinecone.py")
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
