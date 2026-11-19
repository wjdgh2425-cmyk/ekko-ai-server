import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import jwt, JWTError

from sqlalchemy import (
    create_engine,
    String,
    Boolean,
    DateTime,
    Integer,
    ForeignKey,
    select,
    inspect,
    text,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)

from openai import OpenAI


# =========================================================
# 기본 설정
# =========================================================

APP_NAME = "에코 AI 서버"

DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EKKO_BANK_NAME = os.getenv("EKKO_BANK_NAME", "")
EKKO_BANK_ACCOUNT = os.getenv("EKKO_BANK_ACCOUNT", "")
EKKO_BANK_HOLDER = os.getenv("EKKO_BANK_HOLDER", "")

SUBSCRIPTION_PRICE = 9900
SUBSCRIPTION_DAYS = 30
AI_MODEL = "gpt-4o-mini"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 환경 변수가 설정되지 않았습니다.")

# Railway PostgreSQL 주소 호환 처리
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1,
    )

engine_options = {
    "pool_pre_ping": True,
}

if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# =========================================================
# 시간 관련 함수
# =========================================================

def utc_now():
    return datetime.now(timezone.utc)


def make_aware(value):
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def iso(value):
    value = make_aware(value)
    return value.isoformat() if value else None


def month_key(value):
    value = make_aware(value)
    return value.strftime("%Y-%m") if value else None


# =========================================================
# 데이터베이스 모델
# =========================================================

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    password_salt: Mapped[str] = mapped_column(String(255))

    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    payment_status: Mapped[str] = mapped_column(String(30), default="free")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )


class PaymentRequest(Base):
    __tablename__ = "payment_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    depositor_name: Mapped[str] = mapped_column(String(100))
    amount: Mapped[int] = mapped_column(Integer, default=SUBSCRIPTION_PRICE)
    status: Mapped[str] = mapped_column(String(30), default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    payment_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("payment_requests.id"),
        unique=True,
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )


class ChatUsage(Base):
    __tablename__ = "chat_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    model: Mapped[str] = mapped_column(String(100), default=AI_MODEL)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )


class Announcement(Base):
    __tablename__ = "announcements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    title: Mapped[str] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(String(5000))

    is_published: Mapped[bool] = mapped_column(Boolean, default=True)
    is_important: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )


# =========================================================
# 테이블 생성 및 기존 DB 호환 처리
# =========================================================

Base.metadata.create_all(bind=engine)

# 기존 users 테이블에 is_active가 없는 경우에만 추가
try:
    inspector = inspect(engine)
    user_columns = [
        column["name"]
        for column in inspector.get_columns("users")
    ]

    if "is_active" not in user_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE"
                )
            )
except Exception as error:
    print("DB 마이그레이션 확인:", error)


# =========================================================
# FastAPI 및 공통 함수
# =========================================================

app = FastAPI(title=APP_NAME)

security = HTTPBearer(auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str, salt: str):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        310000,
    ).hex()


def create_access_token(user_id: int):
    payload = {
        "sub": str(user_id),
        "exp": utc_now() + timedelta(hours=24),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="로그인이 필요합니다.",
        )

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
        )

        user_id = int(payload.get("sub"))

    except (JWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="로그인 정보가 만료되었거나 올바르지 않습니다.",
        )

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="사용자를 찾을 수 없습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="정지된 계정입니다. 관리자에게 문의해 주세요.",
        )

    return user


def require_admin(
    x_admin_key: str = Header(default=""),
):
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="관리자 키가 서버에 설정되지 않았습니다.",
        )

    if not hmac.compare_digest(
        str(x_admin_key),
        str(ADMIN_API_KEY),
    ):
        raise HTTPException(
            status_code=403,
            detail="관리자 인증에 실패했습니다.",
        )

    return True


def get_subscription_state(db: Session, user: User):
    now = utc_now()

    subscriptions = db.scalars(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    ).all()

    active_subscription = None

    for subscription in subscriptions:
        expires_at = make_aware(subscription.expires_at)

        if expires_at and expires_at > now:
            active_subscription = subscription
            break

    pending_payment = db.scalar(
        select(PaymentRequest)
        .where(
            PaymentRequest.user_id == user.id,
            PaymentRequest.status == "pending",
        )
        .order_by(PaymentRequest.created_at.desc())
    )

    latest_subscription = subscriptions[0] if subscriptions else None

    if active_subscription:
        return {
            "is_premium": True,
            "status": "active",
            "expires_at": iso(active_subscription.expires_at),
            "pending_payment": False,
        }

    if pending_payment:
        return {
            "is_premium": False,
            "status": "pending",
            "expires_at": None,
            "pending_payment": True,
        }

    return {
        "is_premium": False,
        "status": "free",
        "expires_at": (
            iso(latest_subscription.expires_at)
            if latest_subscription
            else None
        ),
        "pending_payment": False,
    }


def user_to_dict(db: Session, user: User):
    subscription = get_subscription_state(db, user)

    return {
        "id": user.id,
        "username": user.username,
        "is_active": user.is_active,
        "is_premium": subscription["is_premium"],
        "payment_status": subscription["status"],
        "expires_at": subscription["expires_at"],
        "created_at": iso(user.created_at),
    }


# =========================================================
# 요청 데이터 모델
# =========================================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class TransferRequestCreate(BaseModel):
    depositor_name: str = Field(min_length=1, max_length=100)


class AdminGrantRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=3650)


class AnnouncementRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=5000)
    is_published: bool = True
    is_important: bool = False


# =========================================================
# 기본 및 상태 확인
# =========================================================

@app.get("/")
def home():
    return {
        "success": True,
        "message": "에코 AI 서버가 정상적으로 실행 중입니다.",
        "service": APP_NAME,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
    }


# =========================================================
# 회원가입 및 로그인
# =========================================================

@app.post("/auth/register")
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    username = request.username.strip()

    if not username:
        raise HTTPException(
            status_code=400,
            detail="아이디를 입력해 주세요.",
        )

    existing_user = db.scalar(
        select(User).where(User.username == username)
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="이미 사용 중인 아이디입니다.",
        )

    salt = secrets.token_hex(16)

    user = User(
        username=username,
        password_salt=salt,
        password_hash=hash_password(request.password, salt),
        is_premium=False,
        payment_status="free",
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)

    return {
        "success": True,
        "message": "회원가입이 완료되었습니다.",
        "token": token,
        "user": user_to_dict(db, user),
    }


@app.post("/auth/login")
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    user = db.scalar(
        select(User).where(User.username == request.username.strip())
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    password_hash = hash_password(
        request.password,
        user.password_salt,
    )

    if not hmac.compare_digest(
        password_hash,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="정지된 계정입니다. 관리자에게 문의해 주세요.",
        )

    token = create_access_token(user.id)

    return {
        "success": True,
        "message": "로그인되었습니다.",
        "token": token,
        "user": user_to_dict(db, user),
    }


@app.get("/me")
def me(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {
        "success": True,
        "user": user_to_dict(db, user),
    }


# =========================================================
# 구독 및 계좌이체
# =========================================================

@app.get("/subscription/info")
def subscription_info(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    state = get_subscription_state(db, user)

    return {
        "success": True,
        "price": SUBSCRIPTION_PRICE,
        "days": SUBSCRIPTION_DAYS,
        "bank_name": EKKO_BANK_NAME,
        "bank_account": EKKO_BANK_ACCOUNT,
        "bank_holder": EKKO_BANK_HOLDER,
        **state,
    }


@app.post("/subscription/transfer-request")
def create_transfer_request(
    request: TransferRequestCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    state = get_subscription_state(db, user)

    if state["is_premium"]:
        raise HTTPException(
            status_code=400,
            detail="이미 프리미엄 구독을 이용 중입니다.",
        )

    if state["pending_payment"]:
        raise HTTPException(
            status_code=400,
            detail="이미 입금 확인 대기 중인 신청이 있습니다.",
        )

    payment = PaymentRequest(
        user_id=user.id,
        depositor_name=request.depositor_name.strip(),
        amount=SUBSCRIPTION_PRICE,
        status="pending",
    )

    db.add(payment)
    db.commit()
    db.refresh(payment)

    return {
        "success": True,
        "message": "구독 신청이 접수되었습니다. 입금 확인 후 승인됩니다.",
        "payment_request_id": payment.id,
        "status": payment.status,
    }


# =========================================================
# 공지사항 - 에코 AI 앱에서 조회
# =========================================================

@app.get("/announcements")
def get_announcements(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    announcements = db.scalars(
        select(Announcement)
        .where(Announcement.is_published == True)
        .order_by(
            Announcement.is_important.desc(),
            Announcement.created_at.desc(),
        )
        .limit(20)
    ).all()

    return {
        "success": True,
        "announcements": [
            {
                "id": item.id,
                "title": item.title,
                "content": item.content,
                "is_important": item.is_important,
                "created_at": iso(item.created_at),
            }
            for item in announcements
        ],
    }


# =========================================================
# AI 채팅 및 실제 토큰 사용량 기록
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    state = get_subscription_state(db, user)

    if not state["is_premium"]:
        raise HTTPException(
            status_code=403,
            detail="프리미엄 구독이 필요한 기능입니다.",
        )

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API 키가 서버에 설정되지 않았습니다.",
        )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.responses.create(
            model=AI_MODEL,
            instructions=(
                "너는 EKKO AI라는 이름의 친절한 AI 비서야. "
                "사용자의 질문에 정확하고 자연스러운 한국어로 답변해 줘."
            ),
            input=request.message,
            max_output_tokens=1200,
        )

        answer = response.output_text

        usage = getattr(response, "usage", None)

        input_tokens = int(
            getattr(usage, "input_tokens", 0) or 0
        )

        output_tokens = int(
            getattr(usage, "output_tokens", 0) or 0
        )

        total_tokens = int(
            getattr(usage, "total_tokens", 0)
            or (input_tokens + output_tokens)
        )

        # 성공한 채팅의 사용량만 저장
        usage_record = ChatUsage(
            user_id=user.id,
            model=AI_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        db.add(usage_record)
        db.commit()

        return {
            "success": True,
            "answer": answer,
            "usage": {
                "model": AI_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print("OpenAI 채팅 오류:", str(error))

        raise HTTPException(
            status_code=500,
            detail="AI 답변을 생성하는 중 오류가 발생했습니다.",
        )


# =========================================================
# 관리자 - 입금 신청 관리
# =========================================================

@app.get("/admin/subscription-requests")
def admin_subscription_requests(
    status: str = "pending",
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = (
        select(PaymentRequest, User)
        .join(User, PaymentRequest.user_id == User.id)
        .order_by(PaymentRequest.created_at.desc())
    )

    if status != "all":
        query = query.where(PaymentRequest.status == status)

    rows = db.execute(query).all()

    return {
        "success": True,
        "requests": [
            {
                "id": payment.id,
                "user_id": user.id,
                "username": user.username,
                "depositor_name": payment.depositor_name,
                "amount": payment.amount,
                "status": payment.status,
                "created_at": iso(payment.created_at),
                "reviewed_at": iso(payment.reviewed_at),
            }
            for payment, user in rows
        ],
    }


def approve_payment(db: Session, payment: PaymentRequest):
    if payment.status != "pending":
        raise HTTPException(
            status_code=400,
            detail="이미 처리된 신청입니다.",
        )

    user = db.get(User, payment.user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    now = utc_now()

    current_subscription = db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.expires_at > now,
        )
        .order_by(Subscription.expires_at.desc())
    )

    if current_subscription:
        start_time = make_aware(current_subscription.expires_at)
    else:
        start_time = now

    subscription = Subscription(
        user_id=user.id,
        payment_request_id=payment.id,
        started_at=now,
        expires_at=start_time + timedelta(days=SUBSCRIPTION_DAYS),
    )

    payment.status = "approved"
    payment.reviewed_at = now

    user.is_premium = True
    user.payment_status = "active"

    db.add(subscription)
    db.commit()

    return {
        "success": True,
        "message": "구독이 승인되었습니다.",
    }


@app.post("/admin/subscription-requests/{payment_id}/approve")
def admin_approve_payment(
    payment_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    payment = db.get(PaymentRequest, payment_id)

    if not payment:
        raise HTTPException(
            status_code=404,
            detail="입금 신청을 찾을 수 없습니다.",
        )

    return approve_payment(db, payment)


@app.post("/admin/subscription-requests/{payment_id}/reject")
def admin_reject_payment(
    payment_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    payment = db.get(PaymentRequest, payment_id)

    if not payment:
        raise HTTPException(
            status_code=404,
            detail="입금 신청을 찾을 수 없습니다.",
        )

    if payment.status != "pending":
        raise HTTPException(
            status_code=400,
            detail="이미 처리된 신청입니다.",
        )

    payment.status = "rejected"
    payment.reviewed_at = utc_now()

    db.commit()

    return {
        "success": True,
        "message": "신청이 거절되었습니다.",
    }


# =========================================================
# 관리자 - 회원 관리
# =========================================================

@app.get("/admin/members")
def admin_members(
    search: str = "",
    status: str = "all",
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = select(User).order_by(User.created_at.desc())

    if search.strip():
        query = query.where(
            User.username.ilike(f"%{search.strip()}%")
        )

    users = db.scalars(query.limit(500)).all()

    result = []

    for user in users:
        state = get_subscription_state(db, user)

        if status == "premium" and not state["is_premium"]:
            continue

        if status == "free" and (
            state["is_premium"] or not user.is_active
        ):
            continue

        if status == "suspended" and user.is_active:
            continue

        result.append({
            **user_to_dict(db, user),
            "is_premium": state["is_premium"],
            "payment_status": state["status"],
        })

    return {
        "success": True,
        "members": result,
        "count": len(result),
    }


@app.get("/admin/members/{user_id}")
def admin_member_detail(
    user_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    payments = db.scalars(
        select(PaymentRequest)
        .where(PaymentRequest.user_id == user_id)
        .order_by(PaymentRequest.created_at.desc())
        .limit(100)
    ).all()

    subscriptions = db.scalars(
        select(Subscription)
        .where(Subscription.user_id == user_id)
        .order_by(Subscription.created_at.desc())
        .limit(100)
    ).all()

    usage_rows = db.scalars(
        select(ChatUsage)
        .where(ChatUsage.user_id == user_id)
        .order_by(ChatUsage.created_at.desc())
        .limit(100)
    ).all()

    usage_summary = db.execute(
        select(
            func.count(ChatUsage.id),
            func.coalesce(func.sum(ChatUsage.input_tokens), 0),
            func.coalesce(func.sum(ChatUsage.output_tokens), 0),
            func.coalesce(func.sum(ChatUsage.total_tokens), 0),
        ).where(ChatUsage.user_id == user_id)
    ).one()

    return {
        "success": True,
        "member": user_to_dict(db, user),
        "payments": [
            {
                "id": item.id,
                "amount": item.amount,
                "depositor_name": item.depositor_name,
                "status": item.status,
                "created_at": iso(item.created_at),
                "reviewed_at": iso(item.reviewed_at),
            }
            for item in payments
        ],
        "subscriptions": [
            {
                "id": item.id,
                "started_at": iso(item.started_at),
                "expires_at": iso(item.expires_at),
                "created_at": iso(item.created_at),
            }
            for item in subscriptions
        ],
        "usage_summary": {
            "chat_count": int(usage_summary[0] or 0),
            "input_tokens": int(usage_summary[1] or 0),
            "output_tokens": int(usage_summary[2] or 0),
            "total_tokens": int(usage_summary[3] or 0),
        },
        "recent_usage": [
            {
                "id": item.id,
                "model": item.model,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
                "created_at": iso(item.created_at),
            }
            for item in usage_rows
        ],
    }


@app.post("/admin/members/{user_id}/grant-subscription")
def admin_grant_subscription(
    user_id: int,
    request: AdminGrantRequest,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=400,
            detail="정지된 회원에게는 구독을 지급할 수 없습니다.",
        )

    now = utc_now()

    current_subscription = db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user_id,
            Subscription.expires_at > now,
        )
        .order_by(Subscription.expires_at.desc())
    )

    if current_subscription:
        start_time = make_aware(current_subscription.expires_at)
    else:
        start_time = now

    # 관리자 지급은 실제 결제가 아니므로 amount=0
    payment = PaymentRequest(
        user_id=user_id,
        depositor_name="관리자 수동 지급",
        amount=0,
        status="approved",
        created_at=now,
        reviewed_at=now,
    )

    db.add(payment)
    db.flush()

    subscription = Subscription(
        user_id=user_id,
        payment_request_id=payment.id,
        started_at=now,
        expires_at=start_time + timedelta(days=request.days),
    )

    user.is_premium = True
    user.payment_status = "active"

    db.add(subscription)
    db.commit()

    return {
        "success": True,
        "message": f"{request.days}일 구독을 지급했습니다.",
    }


@app.post("/admin/members/{user_id}/revoke-subscription")
def admin_revoke_subscription(
    user_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    now = utc_now()

    subscriptions = db.scalars(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.expires_at > now,
        )
    ).all()

    for subscription in subscriptions:
        subscription.expires_at = now

    user.is_premium = False
    user.payment_status = "free"

    db.commit()

    return {
        "success": True,
        "message": "구독이 회수되었습니다.",
    }


@app.post("/admin/members/{user_id}/suspend")
def admin_suspend_member(
    user_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    user.is_active = False
    db.commit()

    return {
        "success": True,
        "message": "회원 계정이 정지되었습니다.",
    }


@app.post("/admin/members/{user_id}/activate")
def admin_activate_member(
    user_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    user.is_active = True
    db.commit()

    return {
        "success": True,
        "message": "회원 계정 정지가 해제되었습니다.",
    }


# =========================================================
# 관리자 - 통계 대시보드 API
# =========================================================

@app.get("/admin/stats")
def admin_stats(
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    now = utc_now()
    today = now.date()

    users = db.scalars(select(User)).all()

    total_members = len(users)
    suspended_members = sum(1 for user in users if not user.is_active)

    premium_members = 0

    for user in users:
        if user.is_active:
            state = get_subscription_state(db, user)
            if state["is_premium"]:
                premium_members += 1

    free_members = max(
        0,
        total_members - suspended_members - premium_members,
    )

    # 최근 7일 가입자
    registration_start = now - timedelta(days=6)

    recent_users = db.scalars(
        select(User).where(
            User.created_at >= registration_start
        )
    ).all()

    daily_registrations = defaultdict(int)

    for offset in range(7):
        day = today - timedelta(days=6 - offset)
        daily_registrations[day.isoformat()] = 0

    for user in recent_users:
        created = make_aware(user.created_at)
        if created:
            key = created.date().isoformat()
            if key in daily_registrations:
                daily_registrations[key] += 1

    # 최근 6개월 매출 및 유료 구독 승인 건수
    six_month_start = now - timedelta(days=183)

    approved_payments = db.scalars(
        select(PaymentRequest).where(
            PaymentRequest.status == "approved",
            PaymentRequest.amount > 0,
        )
    ).all()

    monthly_revenue = defaultdict(int)
    monthly_paid_subscriptions = defaultdict(int)

    for offset in range(6):
        month_date = (
            now.replace(day=1)
            - timedelta(days=offset * 30)
        )

        key = month_key(month_date)
        monthly_revenue[key] = 0
        monthly_paid_subscriptions[key] = 0

    for payment in approved_payments:
        approved_time = make_aware(
            payment.reviewed_at or payment.created_at
        )

        if not approved_time or approved_time < six_month_start:
            continue

        key = month_key(approved_time)

        if key in monthly_revenue:
            monthly_revenue[key] += int(payment.amount or 0)
            monthly_paid_subscriptions[key] += 1

    # 대시보드 월 표시를 시간순으로 정렬
    month_labels = sorted(monthly_revenue.keys())

    pending_count = db.scalar(
        select(func.count(PaymentRequest.id)).where(
            PaymentRequest.status == "pending"
        )
    ) or 0

    usage_summary = db.execute(
        select(
            func.count(ChatUsage.id),
            func.coalesce(func.sum(ChatUsage.input_tokens), 0),
            func.coalesce(func.sum(ChatUsage.output_tokens), 0),
            func.coalesce(func.sum(ChatUsage.total_tokens), 0),
        )
    ).one()

    return {
        "success": True,
        "members": {
            "total": total_members,
            "premium": premium_members,
            "free": free_members,
            "suspended": suspended_members,
        },
        "daily_registrations": [
            {
                "date": key,
                "count": daily_registrations[key],
            }
            for key in sorted(daily_registrations.keys())
        ],
        "monthly_revenue": [
            {
                "month": key,
                "revenue": monthly_revenue[key],
                "paid_subscriptions": monthly_paid_subscriptions[key],
            }
            for key in month_labels
        ],
        "pending_payment_requests": int(pending_count),
        "usage": {
            "chat_count": int(usage_summary[0] or 0),
            "input_tokens": int(usage_summary[1] or 0),
            "output_tokens": int(usage_summary[2] or 0),
            "total_tokens": int(usage_summary[3] or 0),
        },
    }


# =========================================================
# 관리자 - 회원별 사용량 API
# =========================================================

@app.get("/admin/usage")
def admin_usage(
    search: str = "",
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = (
        select(
            User.id,
            User.username,
            func.count(ChatUsage.id).label("chat_count"),
            func.coalesce(
                func.sum(ChatUsage.input_tokens), 0
            ).label("input_tokens"),
            func.coalesce(
                func.sum(ChatUsage.output_tokens), 0
            ).label("output_tokens"),
            func.coalesce(
                func.sum(ChatUsage.total_tokens), 0
            ).label("total_tokens"),
        )
        .outerjoin(ChatUsage, ChatUsage.user_id == User.id)
        .group_by(User.id, User.username)
        .order_by(func.coalesce(func.sum(ChatUsage.total_tokens), 0).desc())
    )

    if search.strip():
        query = query.where(
            User.username.ilike(f"%{search.strip()}%")
        )

    rows = db.execute(query.limit(500)).all()

    return {
        "success": True,
        "usage": [
            {
                "user_id": row.id,
                "username": row.username,
                "chat_count": int(row.chat_count or 0),
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "total_tokens": int(row.total_tokens or 0),
            }
            for row in rows
        ],
    }


# =========================================================
# 관리자 - 공지사항 관리 API
# =========================================================

@app.get("/admin/announcements")
def admin_get_announcements(
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    items = db.scalars(
        select(Announcement)
        .order_by(Announcement.created_at.desc())
        .limit(200)
    ).all()

    return {
        "success": True,
        "announcements": [
            {
                "id": item.id,
                "title": item.title,
                "content": item.content,
                "is_published": item.is_published,
                "is_important": item.is_important,
                "created_at": iso(item.created_at),
                "updated_at": iso(item.updated_at),
            }
            for item in items
        ],
    }


@app.post("/admin/announcements")
def admin_create_announcement(
    request: AnnouncementRequest,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    now = utc_now()

    item = Announcement(
        title=request.title.strip(),
        content=request.content.strip(),
        is_published=request.is_published,
        is_important=request.is_important,
        created_at=now,
        updated_at=now,
    )

    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "success": True,
        "message": "공지사항을 등록했습니다.",
        "id": item.id,
    }


@app.put("/admin/announcements/{announcement_id}")
def admin_update_announcement(
    announcement_id: int,
    request: AnnouncementRequest,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    item = db.get(Announcement, announcement_id)

    if not item:
        raise HTTPException(
            status_code=404,
            detail="공지사항을 찾을 수 없습니다.",
        )

    item.title = request.title.strip()
    item.content = request.content.strip()
    item.is_published = request.is_published
    item.is_important = request.is_important
    item.updated_at = utc_now()

    db.commit()

    return {
        "success": True,
        "message": "공지사항을 수정했습니다.",
    }


@app.delete("/admin/announcements/{announcement_id}")
def admin_delete_announcement(
    announcement_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    item = db.get(Announcement, announcement_id)

    if not item:
        raise HTTPException(
            status_code=404,
            detail="공지사항을 찾을 수 없습니다.",
        )

    db.delete(item)
    db.commit()

    return {
        "success": True,
        "message": "공지사항을 삭제했습니다.",
    }


# =========================================================
# 관리자 웹 페이지
# =========================================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>에코 AI 관리자</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;
    padding:24px;
    background:#090817;
    color:#f4f1ff;
    font-family:Arial,"Malgun Gothic",sans-serif
}
h1{color:#67e8f9;margin-top:0}
h2{font-size:20px;margin-top:0}
button{
    border:0;
    border-radius:8px;
    padding:10px 14px;
    background:#713cff;
    color:white;
    cursor:pointer;
    margin:3px
}
button:hover{filter:brightness(1.15)}
button.red{background:#b83255}
button.gray{background:#34324a}
input,textarea,select{
    width:100%;
    padding:11px;
    margin:6px 0 12px;
    border:1px solid #393451;
    border-radius:8px;
    background:#17152b;
    color:white
}
textarea{min-height:110px;resize:vertical}
.panel{
    background:#151329;
    border:1px solid #302b4b;
    border-radius:14px;
    padding:18px;
    margin:16px 0
}
.tabs{display:flex;flex-wrap:wrap;gap:5px;margin:18px 0}
.tabs button{background:#292541}
.tabs button.active{background:#713cff}
.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
    gap:12px
}
.stat{
    background:#211d3b;
    border-radius:12px;
    padding:16px
}
.stat strong{
    display:block;
    font-size:25px;
    color:#67e8f9;
    margin-top:8px
}
table{width:100%;border-collapse:collapse;min-width:650px}
th,td{
    text-align:left;
    padding:11px;
    border-bottom:1px solid #34304e;
    vertical-align:top
}
th{color:#a9a2c9}
.table-wrap{overflow-x:auto}
.small{font-size:13px;color:#aaa4c5}
.hidden{display:none}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.notice{padding:12px;background:#211d3b;border-radius:8px;margin:8px 0}
@media(max-width:600px){
    body{padding:12px}
    .row{grid-template-columns:1fr}
}
</style>
</head>
<body>

<h1>⚡ 에코 AI 관리자 센터</h1>
<p class="small">회원 · 결제 · 통계 · 공지사항 · AI 사용량 관리</p>

<div class="panel" id="loginPanel">
    <h2>관리자 인증</h2>
    <input id="adminKey" type="password" placeholder="ADMIN_API_KEY 입력">
    <button onclick="login()">로그인</button>
    <p id="loginMessage" class="small"></p>
</div>

<div id="mainPanel" class="hidden">
    <div class="tabs">
        <button class="active" onclick="showTab('stats',this)">통계</button>
        <button onclick="showTab('members',this)">회원 관리</button>
        <button onclick="showTab('payments',this)">입금 신청</button>
        <button onclick="showTab('announcements',this)">공지사항</button>
        <button onclick="showTab('usage',this)">AI 사용량</button>
    </div>

    <section id="stats" class="tab">
        <div class="panel">
            <h2>서비스 통계</h2>
            <button onclick="loadStats()">새로고침</button>
            <div class="grid" id="statCards"></div>
        </div>
        <div class="panel">
            <h2>최근 7일 가입자</h2>
            <div id="dailyStats"></div>
        </div>
        <div class="panel">
            <h2>최근 6개월 매출 및 유료 구독</h2>
            <div id="monthlyStats"></div>
            <p class="small">실제 승인된 결제 금액만 집계합니다. 관리자 무료 지급은 매출에서 제외됩니다.</p>
        </div>
        <div class="panel">
            <h2>AI 사용량 전체</h2>
            <div id="totalUsage"></div>
        </div>
    </section>

    <section id="members" class="tab hidden">
        <div class="panel">
            <h2>회원 관리</h2>
            <div class="row">
                <input id="memberSearch" placeholder="아이디 검색">
                <select id="memberStatus">
                    <option value="all">전체 회원</option>
                    <option value="premium">프리미엄</option>
                    <option value="free">무료 회원</option>
                    <option value="suspended">정지 회원</option>
                </select>
            </div>
            <button onclick="loadMembers()">검색</button>
            <div class="table-wrap" id="memberTable"></div>
        </div>
        <div class="panel hidden" id="memberDetail"></div>
    </section>

    <section id="payments" class="tab hidden">
        <div class="panel">
            <h2>입금 신청 관리</h2>
            <select id="paymentStatus">
                <option value="pending">승인 대기</option>
                <option value="approved">승인 완료</option>
                <option value="rejected">거절</option>
                <option value="all">전체</option>
            </select>
            <button onclick="loadPayments()">조회</button>
            <div class="table-wrap" id="paymentTable"></div>
        </div>
    </section>

    <section id="announcements" class="tab hidden">
        <div class="panel">
            <h2>공지사항 작성</h2>
            <input id="announcementId" type="hidden">
            <label>제목</label>
            <input id="announcementTitle" maxlength="100" placeholder="공지사항 제목">
            <label>내용</label>
            <textarea id="announcementContent" maxlength="5000" placeholder="공지사항 내용을 입력하세요."></textarea>
            <label><input id="announcementPublished" type="checkbox" checked style="width:auto"> 게시하기</label>
            <label><input id="announcementImportant" type="checkbox" style="width:auto"> 중요 공지</label>
            <br>
            <button onclick="saveAnnouncement()">저장</button>
            <button class="gray" onclick="clearAnnouncementForm()">새 공지 작성</button>
        </div>
        <div class="panel">
            <h2>등록된 공지사항</h2>
            <button onclick="loadAnnouncements()">새로고침</button>
            <div id="announcementList"></div>
        </div>
    </section>

    <section id="usage" class="tab hidden">
        <div class="panel">
            <h2>회원별 AI 사용량</h2>
            <input id="usageSearch" placeholder="회원 아이디 검색">
            <button onclick="loadUsage()">조회</button>
            <div class="table-wrap" id="usageTable"></div>
        </div>
    </section>
</div>

<script>
let KEY = "";

async function api(path, method="GET", body=null){
    const options = {
        method,
        headers:{
            "X-Admin-Key":KEY,
            "Content-Type":"application/json"
        }
    };
    if(body !== null) options.body = JSON.stringify(body);

    const response = await fetch(path,options);
    let data = {};
    try{data = await response.json()}catch(e){}

    if(!response.ok){
        throw new Error(data.detail || "요청에 실패했습니다.");
    }
    return data;
}

function money(value){
    return Number(value || 0).toLocaleString("ko-KR") + "원";
}

function number(value){
    return Number(value || 0).toLocaleString("ko-KR");
}

function safe(value){
    return String(value ?? "").replace(/[&<>"']/g, c => ({
        "&":"&amp;",
        "<":"&lt;",
        ">":"&gt;",
        '"':"&quot;",
        "'":"&#39;"
    }[c]));
}

function login(){
    KEY = document.getElementById("adminKey").value.trim();
    if(!KEY){
        document.getElementById("loginMessage").textContent =
            "관리자 키를 입력해 주세요.";
        return;
    }

    document.getElementById("loginMessage").textContent =
        "인증 확인 중...";

    Promise.all([
        api("/admin/stats")
    ]).then(()=>{
        document.getElementById("loginPanel").classList.add("hidden");
        document.getElementById("mainPanel").classList.remove("hidden");
        loadStats();
        loadMembers();
        loadPayments();
        loadAnnouncements();
        loadUsage();
    }).catch(error=>{
        KEY = "";
        document.getElementById("loginMessage").textContent = error.message;
    });
}

function showTab(name,button){
    document.querySelectorAll(".tab").forEach(item=>{
        item.classList.add("hidden");
    });
    document.getElementById(name).classList.remove("hidden");

    document.querySelectorAll(".tabs button").forEach(item=>{
        item.classList.remove("active");
    });
    button.classList.add("active");
}

async function loadStats(){
    try{
        const data = await api("/admin/stats");
        const m = data.members;
        const u = data.usage;

        document.getElementById("statCards").innerHTML = `
            <div class="stat">전체 회원<strong>${number(m.total)}명</strong></div>
            <div class="stat">프리미엄 회원<strong>${number(m.premium)}명</strong></div>
            <div class="stat">무료 회원<strong>${number(m.free)}명</strong></div>
            <div class="stat">정지 회원<strong>${number(m.suspended)}명</strong></div>
            <div class="stat">승인 대기 결제<strong>${number(data.pending_payment_requests)}건</strong></div>
        `;

        document.getElementById("dailyStats").innerHTML =
            data.daily_registrations.map(item=>`
                <div class="notice">
                    ${safe(item.date)} : <b>${number(item.count)}명</b>
                </div>
            `).join("");

        document.getElementById("monthlyStats").innerHTML =
            data.monthly_revenue.map(item=>`
                <div class="notice">
                    ${safe(item.month)} :
                    <b>${money(item.revenue)}</b>
                    <span class="small">
                        · 유료 승인 ${number(item.paid_subscriptions)}건
                    </span>
                </div>
            `).join("");

        document.getElementById("totalUsage").innerHTML = `
            <div class="grid">
                <div class="stat">총 채팅<strong>${number(u.chat_count)}회</strong></div>
                <div class="stat">입력 토큰<strong>${number(u.input_tokens)}</strong></div>
                <div class="stat">출력 토큰<strong>${number(u.output_tokens)}</strong></div>
                <div class="stat">전체 토큰<strong>${number(u.total_tokens)}</strong></div>
            </div>
        `;
    }catch(error){
        alert(error.message);
    }
}

async function loadMembers(){
    try{
        const search = document.getElementById("memberSearch").value;
        const status = document.getElementById("memberStatus").value;

        const data = await api(
            "/admin/members?search="+encodeURIComponent(search)+
            "&status="+encodeURIComponent(status)
        );

        document.getElementById("memberTable").innerHTML = `
            <table>
            <thead><tr>
                <th>ID</th><th>아이디</th><th>상태</th>
                <th>가입일</th><th>관리</th>
            </tr></thead>
            <tbody>
            ${data.members.map(user=>`
                <tr>
                    <td>${user.id}</td>
                    <td>${safe(user.username)}</td>
                    <td>${user.is_active ?
                        (user.is_premium ? "프리미엄" : "무료") :
                        "정지"}
                    </td>
                    <td>${safe(user.created_at || "")}</td>
                    <td>
                        <button onclick="memberDetail(${user.id})">상세</button>
                        ${user.is_active ?
                            `<button class="red" onclick="memberAction(${user.id},'suspend')">정지</button>` :
                            `<button onclick="memberAction(${user.id},'activate')">정지 해제</button>`
                        }
                    </td>
                </tr>
            `).join("")}
            </tbody></table>
        `;
    }catch(error){
        alert(error.message);
    }
}

async function memberAction(id,action){
    const message = action === "suspend" ?
        "이 회원을 정지할까요?" :
        "이 회원의 정지를 해제할까요?";

    if(!confirm(message)) return;

    try{
        await api(`/admin/members/${id}/${action}`,"POST",{});
        await loadMembers();
        alert("처리되었습니다.");
    }catch(error){
        alert(error.message);
    }
}

async function memberDetail(id){
    try{
        const data = await api(`/admin/members/${id}`);
        const user = data.member;
        const u = data.usage_summary;

        document.getElementById("memberDetail").classList.remove("hidden");
        document.getElementById("memberDetail").innerHTML = `
            <h2>회원 상세 정보: ${safe(user.username)}</h2>
            <p>회원 ID: ${user.id}</p>
            <p>계정 상태: ${user.is_active ? "정상" : "정지"}</p>
            <p>구독 상태: ${safe(user.payment_status)}</p>
            <p>채팅 횟수: ${number(u.chat_count)}회</p>
            <p>입력 토큰: ${number(u.input_tokens)}</p>
            <p>출력 토큰: ${number(u.output_tokens)}</p>
            <p>총 토큰: ${number(u.total_tokens)}</p>

            <h3>구독 지급</h3>
            <input id="grantDays" type="number" min="1" max="3650" value="30">
            <button onclick="grantSubscription(${user.id})">구독 지급</button>
            <button class="red" onclick="revokeSubscription(${user.id})">구독 회수</button>

            <h3>최근 사용 기록</h3>
            <div class="table-wrap">
            <table>
            <tr><th>시간</th><th>모델</th><th>입력</th><th>출력</th><th>합계</th></tr>
            ${data.recent_usage.map(item=>`
                <tr>
                    <td>${safe(item.created_at || "")}</td>
                    <td>${safe(item.model)}</td>
                    <td>${number(item.input_tokens)}</td>
                    <td>${number(item.output_tokens)}</td>
                    <td>${number(item.total_tokens)}</td>
                </tr>
            `).join("")}
            </table></div>
        `;
    }catch(error){
        alert(error.message);
    }
}

async function grantSubscription(id){
    const days = Number(document.getElementById("grantDays").value);
    if(!Number.isInteger(days) || days < 1 || days > 3650){
        alert("1일에서 3650일 사이로 입력해 주세요.");
        return;
    }
    if(!confirm(`${days}일 구독을 지급할까요?`)) return;

    try{
        await api(`/admin/members/${id}/grant-subscription`,"POST",{days});
        alert("구독을 지급했습니다.");
        memberDetail(id);
        loadMembers();
        loadStats();
    }catch(error){
        alert(error.message);
    }
}

async function revokeSubscription(id){
    if(!confirm("이 회원의 구독을 회수할까요?")) return;

    try{
        await api(`/admin/members/${id}/revoke-subscription`,"POST",{});
        alert("구독을 회수했습니다.");
        memberDetail(id);
        loadMembers();
        loadStats();
    }catch(error){
        alert(error.message);
    }
}

async function loadPayments(){
    try{
        const status = document.getElementById("paymentStatus").value;
        const data = await api(
            "/admin/subscription-requests?status="+encodeURIComponent(status)
        );

        document.getElementById("paymentTable").innerHTML = `
            <table>
            <tr>
                <th>신청 ID</th><th>회원</th><th>입금자명</th>
                <th>금액</th><th>상태</th><th>신청일</th><th>처리</th>
            </tr>
            ${data.requests.map(item=>`
                <tr>
                    <td>${item.id}</td>
                    <td>${safe(item.username)}</td>
                    <td>${safe(item.depositor_name)}</td>
                    <td>${money(item.amount)}</td>
                    <td>${safe(item.status)}</td>
                    <td>${safe(item.created_at || "")}</td>
                    <td>
                        ${item.status === "pending" ? `
                            <button onclick="paymentAction(${item.id},'approve')">승인</button>
                            <button class="red" onclick="paymentAction(${item.id},'reject')">거절</button>
                        ` : "-"}
                    </td>
                </tr>
            `).join("")}
            </table>
        `;
    }catch(error){
        alert(error.message);
    }
}

async function paymentAction(id,action){
    const label = action === "approve" ? "승인" : "거절";
    if(!confirm(`이 입금 신청을 ${label}할까요?`)) return;

    try{
        await api(
            `/admin/subscription-requests/${id}/${action}`,
            "POST",
            {}
        );
        alert("처리되었습니다.");
        loadPayments();
        loadStats();
        loadMembers();
    }catch(error){
        alert(error.message);
    }
}

function clearAnnouncementForm(){
    document.getElementById("announcementId").value = "";
    document.getElementById("announcementTitle").value = "";
    document.getElementById("announcementContent").value = "";
    document.getElementById("announcementPublished").checked = true;
    document.getElementById("announcementImportant").checked = false;
}

async function loadAnnouncements(){
    try{
        const data = await api("/admin/announcements");

        document.getElementById("announcementList").innerHTML =
            data.announcements.map(item=>`
                <div class="notice">
                    <b>${item.is_important ? "⭐ 중요 · " : ""}${safe(item.title)}</b>
                    <p>${safe(item.content).replace(/\n/g,"<br>")}</p>
                    <p class="small">
                        ${item.is_published ? "게시 중" : "비공개"}
                        · ${safe(item.created_at || "")}
                    </p>
                    <button onclick="editAnnouncement(${item.id})">수정</button>
                    <button class="red" onclick="deleteAnnouncement(${item.id})">삭제</button>
                </div>
            `).join("");
    }catch(error){
        alert(error.message);
    }
}

async function saveAnnouncement(){
    const id = document.getElementById("announcementId").value;
    const body = {
        title:document.getElementById("announcementTitle").value.trim(),
        content:document.getElementById("announcementContent").value.trim(),
        is_published:document.getElementById("announcementPublished").checked,
        is_important:document.getElementById("announcementImportant").checked
    };

    if(!body.title || !body.content){
        alert("제목과 내용을 입력해 주세요.");
        return;
    }

    try{
        if(id){
            await api(`/admin/announcements/${id}`,"PUT",body);
        }else{
            await api("/admin/announcements","POST",body);
        }
        clearAnnouncementForm();
        loadAnnouncements();
        alert("공지사항을 저장했습니다.");
    }catch(error){
        alert(error.message);
    }
}

async function editAnnouncement(id){
    try{
        const data = await api("/admin/announcements");
        const item = data.announcements.find(x=>x.id === id);
        if(!item) return;

        document.getElementById("announcementId").value = item.id;
        document.getElementById("announcementTitle").value = item.title;
        document.getElementById("announcementContent").value = item.content;
        document.getElementById("announcementPublished").checked = item.is_published;
        document.getElementById("announcementImportant").checked = item.is_important;

        window.scrollTo({top:0,behavior:"smooth"});
    }catch(error){
        alert(error.message);
    }
}

async function deleteAnnouncement(id){
    if(!confirm("이 공지사항을 삭제할까요?")) return;

    try{
        await api(`/admin/announcements/${id}`,"DELETE");
        loadAnnouncements();
        alert("삭제했습니다.");
    }catch(error){
        alert(error.message);
    }
}

async function loadUsage(){
    try{
        const search = document.getElementById("usageSearch").value;
        const data = await api(
            "/admin/usage?search="+encodeURIComponent(search)
        );

        document.getElementById("usageTable").innerHTML = `
            <table>
            <tr>
                <th>회원</th><th>채팅</th><th>입력 토큰</th>
                <th>출력 토큰</th><th>전체 토큰</th>
            </tr>
            ${data.usage.map(item=>`
                <tr>
                    <td>
                        <a href="#" onclick="memberDetail(${item.user_id});return false;"
                           style="color:#67e8f9">
                            ${safe(item.username)}
                        </a>
                    </td>
                    <td>${number(item.chat_count)}</td>
                    <td>${number(item.input_tokens)}</td>
                    <td>${number(item.output_tokens)}</td>
                    <td>${number(item.total_tokens)}</td>
                </tr>
            `).join("")}
            </table>
        `;
    }catch(error){
        alert(error.message);
    }
}
</script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return ADMIN_HTML
