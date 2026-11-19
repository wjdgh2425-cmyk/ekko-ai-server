
import os
import secrets
import hashlib
import hmac

from datetime import datetime, timedelta, timezone

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
)
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
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)

from openai import OpenAI


# =====================================
# 1. 환경 변수
# =====================================

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")

# 관리자 인증용 비밀키
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

# 계좌이체 안내 정보
BANK_NAME = os.environ.get("EKKO_BANK_NAME", "")
BANK_ACCOUNT = os.environ.get("EKKO_BANK_ACCOUNT", "")
BANK_HOLDER = os.environ.get("EKKO_BANK_HOLDER", "")

# 구독 정책
SUBSCRIPTION_PRICE = 9900
SUBSCRIPTION_DAYS = 30

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 환경 변수가 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")


# Railway PostgreSQL 주소 호환
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )


# =====================================
# 2. 데이터베이스 연결
# =====================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# =====================================
# 3. 사용자 테이블
# =====================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    username: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    password_salt: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    is_premium: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    payment_status: Mapped[str] = mapped_column(
        String(20),
        default="none",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# =====================================
# 4. 계좌이체 신청 테이블
# =====================================

class PaymentRequest(Base):
    __tablename__ = "payment_requests"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    depositor_name: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    amount: Mapped[int] = mapped_column(
        Integer,
        default=SUBSCRIPTION_PRICE,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# =====================================
# 5. 구독 기간 테이블
# =====================================

class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    payment_request_id: Mapped[int] = mapped_column(
        ForeignKey("payment_requests.id"),
        nullable=False,
        unique=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# 새 테이블 생성
# 기존 users 테이블의 데이터는 유지됩니다.
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================
# 6. 구독 상태 확인
# =====================================

def get_subscription_state(db: Session, user: User):

    now = datetime.now(timezone.utc)

    active_subscription = db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.expires_at > now,
        )
        .order_by(Subscription.expires_at.desc())
    )

    latest_subscription = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    pending_request = db.scalar(
        select(PaymentRequest)
        .where(
            PaymentRequest.user_id == user.id,
            PaymentRequest.status == "pending",
        )
        .order_by(PaymentRequest.created_at.desc())
    )

    if active_subscription:
        status = "active"
    elif pending_request:
        status = "pending"
    elif latest_subscription:
        status = "expired"
    else:
        status = "none"

    return {
        "is_premium": active_subscription is not None,
        "payment_status": status,
        "subscription": active_subscription,
        "pending_request": pending_request,
    }


def require_active_subscription(
    user: User,
    db: Session,
):
    state = get_subscription_state(db, user)

    if not state["is_premium"]:
        raise HTTPException(
            status_code=403,
            detail="EKKO AI PREMIUM 구독이 필요합니다.",
        )

    return state["subscription"]


# =====================================
# 7. 비밀번호 보안
# =====================================

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000,
    ).hex()


def create_password(password: str):
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    return password_hash, salt


def verify_password(password: str, user: User) -> bool:
    check_hash = hash_password(
        password,
        user.password_salt,
    )

    return hmac.compare_digest(
        check_hash,
        user.password_hash,
    )


# =====================================
# 8. JWT 로그인 인증
# =====================================

security = HTTPBearer()

TOKEN_EXPIRE_MINUTES = 60 * 24


def create_access_token(user_id: int):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
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

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
        )

        user_id = int(payload.get("sub"))

    except (JWTError, TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="로그인이 만료되었거나 유효하지 않습니다.",
        )

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="사용자를 찾을 수 없습니다.",
        )

    return user


# =====================================
# 9. 관리자 인증
# =====================================

def require_admin(
    admin_key: str | None = Header(
        default=None,
        alias="X-Admin-Key",
    ),
):

    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="관리자 인증 키가 설정되지 않았습니다.",
        )

    if not admin_key or not hmac.compare_digest(
        admin_key,
        ADMIN_API_KEY,
    ):
        raise HTTPException(
            status_code=403,
            detail="관리자 인증에 실패했습니다.",
        )

    return True


# =====================================
# 10. 요청 데이터
# =====================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class TransferRequestCreate(BaseModel):
    depositor_name: str = Field(
        min_length=2,
        max_length=40,
    )


# =====================================
# 11. FastAPI 서버
# =====================================

app = FastAPI(
    title="EKKO AI Server",
    description="EKKO AI 공식 백엔드 서버",
    version="2.0.0",
)


@app.get("/")
def home():
    return {
        "message": "EKKO AI 서버가 정상 작동 중입니다.",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "EKKO AI",
    }


# =====================================
# 12. 회원가입
# =====================================

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
            status_code=409,
            detail="이미 사용 중인 아이디입니다.",
        )

    password_hash, salt = create_password(
        request.password
    )

    user = User(
        username=username,
        password_hash=password_hash,
        password_salt=salt,
        is_premium=False,
        payment_status="none",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "message": "회원가입이 완료되었습니다.",
        "username": user.username,
    }


# =====================================
# 13. 로그인
# =====================================

@app.post("/auth/login")
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):

    user = db.scalar(
        select(User).where(
            User.username == request.username.strip()
        )
    )

    if not user or not verify_password(
        request.password,
        user,
    ):
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    access_token = create_access_token(user.id)

    state = get_subscription_state(db, user)

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "is_premium": state["is_premium"],
    }


# =====================================
# 14. 내 계정 정보
# =====================================

@app.get("/me")
def my_account(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    state = get_subscription_state(db, user)

    subscription = state["subscription"]

    return {
        "id": user.id,
        "username": user.username,
        "is_premium": state["is_premium"],
        "payment_status": state["payment_status"],
        "subscription_expires_at": (
            subscription.expires_at.isoformat()
            if subscription else None
        ),
        "created_at": user.created_at.isoformat(),
    }


# =====================================
# 15. 구독 안내 정보
# =====================================

@app.get("/subscription/info")
def subscription_info(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    state = get_subscription_state(db, user)

    pending = state["pending_request"]
    subscription = state["subscription"]

    return {
        "success": True,
        "price": SUBSCRIPTION_PRICE,
        "duration_days": SUBSCRIPTION_DAYS,
        "bank_name": BANK_NAME,
        "bank_account": BANK_ACCOUNT,
        "bank_holder": BANK_HOLDER,
        "is_premium": state["is_premium"],
        "payment_status": state["payment_status"],
        "pending_request_id": pending.id if pending else None,
        "subscription_expires_at": (
            subscription.expires_at.isoformat()
            if subscription else None
        ),
    }


# =====================================
# 16. 계좌이체 신청
# =====================================

@app.post("/subscription/transfer-request")
def create_transfer_request(
    request: TransferRequestCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    if not BANK_NAME or not BANK_ACCOUNT or not BANK_HOLDER:
        raise HTTPException(
            status_code=503,
            detail="계좌 정보가 아직 설정되지 않았습니다.",
        )

    state = get_subscription_state(db, user)

    if state["is_premium"]:
        raise HTTPException(
            status_code=409,
            detail="이미 프리미엄 구독을 이용 중입니다.",
        )

    if state["pending_request"]:
        raise HTTPException(
            status_code=409,
            detail="이미 입금 확인 대기 중인 신청이 있습니다.",
        )

    payment_request = PaymentRequest(
        user_id=user.id,
        depositor_name=request.depositor_name.strip(),
        amount=SUBSCRIPTION_PRICE,
        status="pending",
    )

    if not payment_request.depositor_name:
        raise HTTPException(
            status_code=400,
            detail="입금자명을 입력해 주세요.",
        )

    db.add(payment_request)
    db.commit()
    db.refresh(payment_request)

    return {
        "success": True,
        "message": "구독 신청이 완료되었습니다. 입금 확인을 기다려 주세요.",
        "request_id": payment_request.id,
        "status": payment_request.status,
        "amount": payment_request.amount,
    }


# =====================================
# 17. 관리자 - 입금 신청 목록
# =====================================

@app.get("/admin/subscription-requests")
def admin_subscription_requests(
    status: str = "pending",
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    if status not in ["pending", "approved", "rejected", "all"]:
        raise HTTPException(
            status_code=400,
            detail="올바르지 않은 상태입니다.",
        )

    query = (
        select(PaymentRequest, User)
        .join(User, PaymentRequest.user_id == User.id)
        .order_by(PaymentRequest.created_at.desc())
        .limit(100)
    )

    if status != "all":
        query = query.where(
            PaymentRequest.status == status
        )

    results = db.execute(query).all()

    requests = []

    for payment_request, user in results:
        requests.append({
            "request_id": payment_request.id,
            "user_id": user.id,
            "username": user.username,
            "depositor_name": payment_request.depositor_name,
            "amount": payment_request.amount,
            "status": payment_request.status,
            "created_at": payment_request.created_at.isoformat(),
        })

    return {
        "success": True,
        "requests": requests,
    }


# =====================================
# 18. 관리자 - 입금 승인
# =====================================

@app.post(
    "/admin/subscription-requests/{request_id}/approve"
)
def approve_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(
        PaymentRequest,
        request_id,
    )

    if not payment_request:
        raise HTTPException(
            status_code=404,
            detail="구독 신청을 찾을 수 없습니다.",
        )

    if payment_request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="이미 처리된 구독 신청입니다.",
        )

    user = db.get(
        User,
        payment_request.user_id,
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

    # 기존 구독이 남아 있다면 그 만료일 다음부터 연장
    now = datetime.now(timezone.utc)

    latest_subscription = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    if latest_subscription:
        previous_expiration = latest_subscription.expires_at

        if previous_expiration.tzinfo is None:
            previous_expiration = previous_expiration.replace(
                tzinfo=timezone.utc
            )

        started_at = max(now, previous_expiration)
    else:
        started_at = now

    expires_at = started_at + timedelta(
        days=SUBSCRIPTION_DAYS
    )

    subscription = Subscription(
        user_id=user.id,
        payment_request_id=payment_request.id,
        started_at=started_at,
        expires_at=expires_at,
    )

    payment_request.status = "approved"
    payment_request.reviewed_at = now

    # 기존 앱과의 호환성을 위한 표시값
    user.is_premium = True
    user.payment_status = "active"

    db.add(subscription)
    db.commit()

    return {
        "success": True,
        "message": "구독 승인이 완료되었습니다.",
        "username": user.username,
        "started_at": started_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


# =====================================
# 19. 관리자 - 입금 신청 거절
# =====================================

@app.post(
    "/admin/subscription-requests/{request_id}/reject"
)
def reject_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(
        PaymentRequest,
        request_id,
    )

    if not payment_request:
        raise HTTPException(
            status_code=404,
            detail="구독 신청을 찾을 수 없습니다.",
        )

    if payment_request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="이미 처리된 구독 신청입니다.",
        )

    payment_request.status = "rejected"
    payment_request.reviewed_at = datetime.now(
        timezone.utc
    )

    db.commit()

    return {
        "success": True,
        "message": "구독 신청이 거절되었습니다.",
        "request_id": payment_request.id,
    }


# =====================================
# 20. EKKO AI 채팅
# =====================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    # 실제 구독 기간이 남아 있는 사용자만 이용 가능
    require_active_subscription(user, db)

    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                "너는 EKKO AI라는 이름의 친절한 AI 비서야. "
                "사용자의 질문에 정확하고 자연스러운 한국어로 답변해 줘."
            ),
            input=request.message,
            max_output_tokens=1200,
        )

        return {
            "success": True,
            "answer": response.output_text,
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI 서버와 통신하는 중 오류가 발생했습니다.",
        )
