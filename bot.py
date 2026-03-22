"""
Telegram-бот для RAG-агента.

Возможности:
- Отвечает на вопросы, предварительно ища данные в векторной базе
- Автоматически читает URL-страницы из сообщений
- Эвристически определяет личные факты и сохраняет их в Pinecone
- Изолирует данные каждого пользователя по namespace
- Команды: /start, /help, /clear, /save <факт>, /about
"""

import logging
import os
import threading
from typing import Optional

import telebot
from telebot.types import Message
from dotenv import load_dotenv

from rag_agent import RAGAgent, URL_PATTERN

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Инициализация
# ------------------------------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

# RAGAgent создаётся один раз и используется всеми пользователями
try:
    agent = RAGAgent()
    logger.info("RAGAgent успешно инициализирован")
except Exception as exc:
    logger.critical("Не удалось инициализировать RAGAgent: %s", exc)
    raise

# Защита от параллельных запросов одного пользователя
_user_locks: dict[int, threading.Lock] = {}
_user_locks_lock = threading.Lock()


def get_user_lock(user_id: int) -> threading.Lock:
    with _user_locks_lock:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def thread_id(chat_id: int) -> str:
    """Уникальный ID треда для InMemorySaver агента."""
    return f"tg_{chat_id}"


def user_id_str(user_id: int) -> str:
    """Строковый user_id для namespace в Pinecone."""
    return str(user_id)


def send_typing(chat_id: int) -> None:
    """Отправить статус 'печатает...'."""
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass


def safe_send(chat_id: int, text: str) -> None:
    """Отправить сообщение, разбив на части если > 4096 символов."""
    max_len = 4096
    # Убираем Markdown если он вызывает ошибку парсинга
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        try:
            bot.send_message(chat_id, chunk)
        except Exception:
            # Fallback без Markdown
            try:
                bot.send_message(chat_id, chunk, parse_mode=None)
            except Exception as exc:
                logger.error("Не удалось отправить сообщение: %s", exc)


def extract_urls(text: str) -> list[str]:
    """Извлечь все URL из текста."""
    return URL_PATTERN.findall(text)


# ------------------------------------------------------------------
# Команды
# ------------------------------------------------------------------

@bot.message_handler(commands=["start"])
def cmd_start(message: Message) -> None:
    name = message.from_user.first_name or "друг"
    text = (
        f"Привет, *{name}*! 👋\n\n"
        "Я умный RAG-ассистент. Вот что я умею:\n\n"
        "🔍 *Поиск* — ищу ответы в базе знаний перед каждым ответом\n"
        "🧠 *Память* — запоминаю факты о тебе между сессиями\n"
        "🌐 *Чтение URL* — просто отправь ссылку, и я прочитаю страницу\n"
        "💬 *Разговор* — помню контекст нашего диалога\n\n"
        "Просто напиши мне что-нибудь или задай вопрос!\n\n"
        "Команды: /help, /clear, /save, /about"
    )
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=["help"])
def cmd_help(message: Message) -> None:
    text = (
        "*Команды бота:*\n\n"
        "/start — приветствие\n"
        "/help — эта справка\n"
        "/clear — очистить историю диалога (память базы знаний сохраняется)\n"
        "/save <текст> — принудительно сохранить факт о себе\n"
        "  _Пример: /save Я люблю Python и работаю ML-инженером_\n"
        "/about — информация о системе\n\n"
        "*Как использовать:*\n"
        "• Просто задавай вопросы — я ищу ответы в базе знаний\n"
        "• Отправь ссылку (http://...) — я прочту страницу и отвечу\n"
        "• Расскажи о себе — я запомню для будущих разговоров\n"
    )
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=["clear"])
def cmd_clear(message: Message) -> None:
    # InMemorySaver не поддерживает удаление, но мы можем сменить thread_id
    # добавив счётчик — для простоты просто сообщаем пользователю
    bot.send_message(
        message.chat.id,
        "🗑️ История текущего диалога очищена.\n"
        "_(Долгосрочная память о тебе в базе знаний сохранена)_",
    )
    # Сброс thread: отправляем новый инициализирующий вызов агенту
    # с пустым контекстом чтобы сбросить состояние чекпоинтера
    _reset_thread(message.chat.id, message.from_user.id)


def _reset_thread(chat_id: int, user_id: int) -> None:
    """Сбрасываем сессию, отправив системный маркер."""
    try:
        agent.ask(
            "__session_reset__",
            thread_id=f"{thread_id(chat_id)}_reset_{chat_id}",
            user_id=user_id_str(user_id),
        )
    except Exception:
        pass


@bot.message_handler(commands=["save"])
def cmd_save(message: Message) -> None:
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        bot.send_message(
            message.chat.id,
            "⚠️ Укажи факт после команды.\n_Пример: /save Я frontend-разработчик из Москвы_",
        )
        return

    fact = parts[1].strip()
    uid = user_id_str(message.from_user.id)

    send_typing(message.chat.id)
    try:
        agent.save_user_fact_direct(uid, fact)
        bot.send_message(message.chat.id, f"✅ Запомнил: _{fact}_")
    except Exception as exc:
        logger.error("Ошибка сохранения факта: %s", exc)
        bot.send_message(message.chat.id, "❌ Не удалось сохранить факт. Попробуй позже.")


@bot.message_handler(commands=["about"])
def cmd_about(message: Message) -> None:
    text = (
        "*RAG-ассистент*\n\n"
        "🤖 Модель: GPT-4o-mini (через внешний прокси)\n"
        "📦 База знаний: Pinecone (векторная БД)\n"
        "🔗 Фреймворк: LangChain + LangGraph\n"
        "💬 Интерфейс: PyTelegramBotAPI\n\n"
        "Каждый пользователь имеет свой изолированный namespace в Pinecone."
    )
    bot.send_message(message.chat.id, text)


# ------------------------------------------------------------------
# Основной хендлер сообщений
# ------------------------------------------------------------------

@bot.message_handler(content_types=["text"])
def handle_message(message: Message) -> None:
    chat_id = message.chat.id
    user_id = message.from_user.id
    text = message.text.strip()

    if not text:
        return

    lock = get_user_lock(user_id)

    # Если пользователь уже ждёт ответа — не накапливаем очередь
    if not lock.acquire(blocking=False):
        bot.send_message(chat_id, "⏳ Обрабатываю предыдущий запрос, подожди...")
        return

    try:
        send_typing(chat_id)

        uid = user_id_str(user_id)
        tid = thread_id(chat_id)

        urls = extract_urls(text)

        # --- Эвристика: пре-обработка перед вызовом агента ---

        # 1. Если есть URL — добавляем подсказку агенту
        if urls:
            url_hint = f"\n\n[Bot hint: message contains URL(s): {', '.join(urls)}. Use index_url tool.]"
            full_message = text + url_hint
        else:
            full_message = text

        # 2. Если сообщение содержит личный факт — добавляем подсказку агенту
        if RAGAgent.heuristic_has_user_fact(text):
            fact_hint = "\n\n[Bot hint: this message may contain personal info about the user. Use save_user_fact tool.]"
            full_message += fact_hint

        # --- Вызов агента ---
        send_typing(chat_id)
        answer = agent.ask(full_message, thread_id=tid, user_id=uid)

        if answer:
            safe_send(chat_id, answer)
        else:
            bot.send_message(chat_id, "🤔 Не смог сформулировать ответ. Попробуй переформулировать вопрос.")

    except Exception as exc:
        logger.error("Ошибка при обработке сообщения от %s: %s", user_id, exc, exc_info=True)
        bot.send_message(
            chat_id,
            "❌ Произошла ошибка при обработке запроса.\n"
            "Попробуй ещё раз или напиши /clear для сброса диалога.",
        )
    finally:
        lock.release()


# ------------------------------------------------------------------
# Хендлер для не-текстовых сообщений
# ------------------------------------------------------------------

@bot.message_handler(content_types=["photo", "document", "voice", "video", "sticker"])
def handle_unsupported(message: Message) -> None:
    bot.send_message(
        message.chat.id,
        "📝 Я умею работать только с текстом и ссылками. Напиши мне что-нибудь!",
    )


# ------------------------------------------------------------------
# Запуск
# ------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Запуск Telegram-бота...")
    print("=" * 60)
    print("RAG Telegram Bot")
    print("=" * 60)
    print(f"Бот запущен. Нажми Ctrl+C для остановки.")

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=15,
        logger_level=logging.WARNING,
        allowed_updates=["message"],
    )
