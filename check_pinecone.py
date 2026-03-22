"""
Диагностика подключения к Pinecone и проверка записи.
Запуск: python check_pinecone.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 60)
    print("Диагностика Pinecone")
    print("=" * 60)

    # 1. Проверка переменных
    pc_key = os.environ.get("PINECONE_API_KEY")
    pc_index = os.environ.get("PINECONE_INDEX_NAME")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_BASE_URL")

    print(f"\n1. Переменные окружения:")
    print(f"   PINECONE_API_KEY: {'***' + pc_key[-4:] if pc_key else 'НЕ ЗАДАНА'}")
    print(f"   PINECONE_INDEX_NAME: {pc_index or 'НЕ ЗАДАН'}")
    print(f"   OPENAI_API_KEY: {'***' + openai_key[-4:] if openai_key else 'НЕ ЗАДАНА'}")
    print(f"   OPENAI_BASE_URL: {openai_url or '(не задан, будет использован OpenAI)'}")

    if not all([pc_key, pc_index, openai_key]):
        print("\n[ОШИБКА] Заполните .env (см. .env.example)")
        return 1

    # 2. Подключение к Pinecone
    print("\n2. Подключение к Pinecone...")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(pc_index)
        stats = index.describe_index_stats()
        print(f"   [OK] Подключение успешно")
        print(f"   dimension: {stats.dimension}")
        print(f"   total_vector_count: {stats.total_vector_count}")
        print(f"   namespaces: {stats.namespaces or '(пусто)'}")
    except Exception as e:
        print(f"   [ОШИБКА] {e}")
        return 1

    # 3. Проверка эмбеддингов
    print("\n3. Проверка эмбеддингов (OpenAI)...")
    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key,
            base_url=openai_url,
        )
        vec = emb.embed_query("test")
        dim = len(vec)
        print(f"   [OK] Размерность: {dim}")
        if stats.dimension and stats.dimension != dim:
            print(f"   [КРИТИЧНО] Индекс ожидает dimension={stats.dimension}, эмбеддинги дают {dim}")
            print(f"   Решение: создай новый индекс с dimension={dim} или смени embedding-модель на 512-мерную.")
    except Exception as e:
        print(f"   [ОШИБКА] {e}")
        return 1

    # 4. Тест записи через langchain-pinecone
    print("\n4. Тест записи в Pinecone (namespace: _diagnostic)...")
    try:
        from langchain_pinecone import Pinecone as PineconeVectorStore
        from langchain_core.documents import Document

        store = PineconeVectorStore(
            embedding=emb,
            index=index,
            text_key="text",
            namespace="_diagnostic",
        )
        doc = Document(page_content="Тестовый документ для проверки записи в Pinecone.", metadata={"source": "check_pinecone"})
        ids = store.add_documents([doc], namespace="_diagnostic")
        print(f"   [OK] Записано. ID: {ids}")
    except Exception as e:
        print(f"   [ОШИБКА] {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 5. Проверка чтения
    print("\n5. Проверка чтения (similarity_search)...")
    try:
        results = store.similarity_search("тестовый документ", k=1, namespace="_diagnostic")
        if results:
            print(f"   [OK] Найдено: {results[0].page_content[:60]}...")
        else:
            print("   [ВНИМАНИЕ] Запись прошла, но поиск ничего не вернул (подожди 1-2 минуты — eventual consistency)")
    except Exception as e:
        print(f"   [ОШИБКА] {e}")

    # 6. Повторная статистика
    print("\n6. Статистика после записи...")
    try:
        stats2 = index.describe_index_stats()
        ns = stats2.namespaces or {}
        diag = ns.get("_diagnostic", {})
        print(f"   namespace _diagnostic: vector_count={diag.get('vector_count', 0)}")
    except Exception as e:
        print(f"   [ОШИБКА] {e}")

    print("\n" + "=" * 60)
    print("Диагностика завершена.")
    return 0


if __name__ == "__main__":
    exit(main())
