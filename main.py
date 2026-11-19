
import os
import secrets
import hashlib
import hmac

from datetime import datetime, timedelta, timezone

from fastapi import (
    FastAPI, Depends, HTTPException, Header
)
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jose import jwt, JWTError

from sqlalchemy import (
    create_engine, String, Boolean, DateTime,
    Integer, ForeignKey, select
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
    sessionmaker, Session
)

from openai import OpenAI


# =====================================
# 1. 환경 변수
# =====================================

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

BANK_NAME = os.environ.get("EKKO_BANK_NAME", "")
BANK_ACCOUNT = os.environ.get("EKKO_BANK_ACCOUNT", "")
BANK_HOLDER = os.environ.get("EKKO_BANK_HOLDER", "")

SUBSCRIPTION_PRICE = 9900
SUBSCRIPTION_DAYS = 30

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 환경 변수가 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )


# =====================================
# 2. 데이터베이스
# =====================================

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(30), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    is_premium: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    payment_status: Mapped[str] = mapped_column(
        String(20), default="none", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PaymentRequest(Base):
    __tablename__ = "payment_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    depositor_name: Mapped[str] = mapped_column(String(40), nullable=False)
    amount: Mapped[int] = mapped_column(
        Integer, default=SUBSCRIPTION_PRICE, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    payment_request_id: Mapped[int] = mapped_column(
        ForeignKey("payment_requests.id"), nullable=False, unique=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================
# 3. 구독 상태
# =====================================

def get_subscription_state(db: Session, user: User):

    now = datetime.now(timezone.utc)

    active = db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.expires_at > now,
        )
        .order_by(Subscription.expires_at.desc())
    )

    latest = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    pending = db.scalar(
        select(PaymentRequest)
        .where(
            PaymentRequest.user_id == user.id,
            PaymentRequest.status == "pending",
        )
        .order_by(PaymentRequest.created_at.desc())
    )

    if active:
        status = "active"
    elif pending:
        status = "pending"
    elif latest:
        status = "expired"
    else:
        status = "none"

    return {
        "is_premium": active is not None,
        "payment_status": status,
        "subscription": active,
        "pending_request": pending,
    }


def require_active_subscription(user: User, db: Session):

    state = get_subscription_state(db, user)

    if not state["is_premium"]:
        raise HTTPException(
            status_code=403,
            detail="EKKO AI PREMIUM 구독이 필요합니다.",
        )

    return state["subscription"]


# =====================================
# 4. 비밀번호
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
    return hash_password(password, salt), salt


def verify_password(password: str, user: User) -> bool:
    check_hash = hash_password(password, user.password_salt)
    return hmac.compare_digest(check_hash, user.password_hash)


# =====================================
# 5. JWT 인증
# =====================================

security = HTTPBearer()
TOKEN_EXPIRE_MINUTES = 60 * 24


def create_access_token(user_id: int):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        JWT_SECRET,
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):

    try:
        payload = jwt.decode(
            credentials.credentials,
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
# 6. 관리자 인증
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
        admin_key, ADMIN_API_KEY
    ):
        raise HTTPException(
            status_code=403,
            detail="관리자 인증에 실패했습니다.",
        )

    return True


# =====================================
# 7. 요청 모델
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
    depositor_name: str = Field(min_length=2, max_length=40)


# =====================================
# 8. FastAPI
# =====================================

app = FastAPI(
    title="EKKO AI Server",
    description="EKKO AI 공식 백엔드 서버",
    version="3.0.0",
)


@app.get("/")
def home():
    return {
        "message": "EKKO AI 서버가 정상 작동 중입니다.",
        "version": "3.0.0",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "EKKO AI"}


# =====================================
# 9. 회원가입
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

    existing = db.scalar(
        select(User).where(User.username == username)
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="이미 사용 중인 아이디입니다.",
        )

    password_hash, salt = create_password(request.password)

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
# 10. 로그인
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

    if not user or not verify_password(request.password, user):
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    state = get_subscription_state(db, user)

    return {
        "success": True,
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "username": user.username,
        "is_premium": state["is_premium"],
    }


# =====================================
# 11. 내 계정
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
# 12. 구독 정보
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
# 13. 계좌이체 신청
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

    depositor_name = request.depositor_name.strip()

    if not depositor_name:
        raise HTTPException(
            status_code=400,
            detail="입금자명을 입력해 주세요.",
        )

    payment_request = PaymentRequest(
        user_id=user.id,
        depositor_name=depositor_name,
        amount=SUBSCRIPTION_PRICE,
        status="pending",
    )

    db.add(payment_request)
    db.commit()
    db.refresh(payment_request)

    return {
        "success": True,
        "message": "구독 신청이 완료되었습니다.",
        "request_id": payment_request.id,
        "status": payment_request.status,
        "amount": payment_request.amount,
    }


# =====================================
# 14. 관리자 API - 신청 목록
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
        query = query.where(PaymentRequest.status == status)

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
            "reviewed_at": (
                payment_request.reviewed_at.isoformat()
                if payment_request.reviewed_at else None
            ),
        })

    return {
        "success": True,
        "requests": requests,
    }


# =====================================
# 15. 관리자 API - 승인
# =====================================

@app.post("/admin/subscription-requests/{request_id}/approve")
def approve_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(PaymentRequest, request_id)

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

    user = db.get(User, payment_request.user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

    now = datetime.now(timezone.utc)

    latest = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    if latest:

        previous_expiration = latest.expires_at

        if previous_expiration.tzinfo is None:
            previous_expiration = previous_expiration.replace(
                tzinfo=timezone.utc
            )

        started_at = max(now, previous_expiration)

    else:
        started_at = now

    expires_at = started_at + timedelta(days=SUBSCRIPTION_DAYS)

    subscription = Subscription(
        user_id=user.id,
        payment_request_id=payment_request.id,
        started_at=started_at,
        expires_at=expires_at,
    )

    payment_request.status = "approved"
    payment_request.reviewed_at = now

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
# 16. 관리자 API - 거절
# =====================================

@app.post("/admin/subscription-requests/{request_id}/reject")
def reject_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(PaymentRequest, request_id)

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
    payment_request.reviewed_at = datetime.now(timezone.utc)

    db.commit()

    return {
        "success": True,
        "message": "구독 신청이 거절되었습니다.",
        "request_id": payment_request.id,
    }


# =====================================
# 17. 관리자 홈페이지
# =====================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EKKO AI 관리자 센터</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #090912;
    color: #f4f0ff;
    font-family: Arial, "Malgun Gothic", sans-serif;
}

header {
    padding: 22px 35px;
    background: #10101b;
    border-bottom: 1px solid #332544;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    color: #a56cff;
    font-size: 23px;
    font-weight: bold;
    letter-spacing: 2px;
}

.tag {
    color: #36e6dc;
    font-size: 12px;
}

main {
    max-width: 1150px;
    margin: 35px auto;
    padding: 0 20px;
}

h1 { font-size: 28px; }

.muted { color: #9992aa; }

.panel {
    background: #151522;
    border: 1px solid #332544;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

input, select {
    width: 100%;
    padding: 13px;
    border-radius: 8px;
    border: 1px solid #51406d;
    background: #0e0d18;
    color: white;
    margin: 8px 0 15px;
}

button {
    border: 0;
    border-radius: 8px;
    padding: 11px 16px;
    color: white;
    font-weight: bold;
    cursor: pointer;
    background: #7544b6;
}

button:hover { opacity: .85; }

button:disabled {
    opacity: .5;
    cursor: not-allowed;
}

.secondary { background: #30243f; }
.approve { background: #167d68; }
.reject { background: #9c3654; }

.cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

.stat {
    background: #151522;
    border: 1px solid #332544;
    border-radius: 12px;
    padding: 20px;
}

.stat small { color: #9992aa; }
.stat strong {
    display: block;
    font-size: 28px;
    margin-top: 9px;
}

.table-wrap { overflow-x: auto; }

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 760px;
}

th, td {
    text-align: left;
    padding: 14px 10px;
    border-bottom: 1px solid #332544;
    font-size: 13px;
}

th { color: #a99bbd; }

.status {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    background: #493b24;
    color: #ffd47b;
    font-size: 11px;
}

.status.approved {
    background: #173f35;
    color: #69e6bc;
}

.status.rejected {
    background: #492332;
    color: #ff9bb3;
}

.actions {
    display: flex;
    gap: 7px;
}

.hidden { display: none; }

#message {
    margin-top: 12px;
    color: #36e6dc;
    white-space: pre-wrap;
}

@media(max-width: 700px) {
    header { padding: 18px; }
    main { margin: 20px auto; }
    .cards { grid-template-columns: 1fr; }
    .panel { padding: 17px; }
}
</style>
</head>

<body>

<header>
    <div class="logo">EKKO AI ADMIN</div>
    <div class="tag">관리자 전용 시스템</div>
</header>

<main>

<section id="loginPanel" class="panel">
    <h1>관리자 로그인</h1>
    <p class="muted">관리자 인증 키를 입력해 주세요.</p>

    <label>관리자 비밀키</label>
    <input
        id="adminKey"
        type="password"
        placeholder="ADMIN_API_KEY"
        autocomplete="off"
    >

    <button onclick="login()">로그인</button>
    <div id="loginMessage"></div>
</section>

<section id="dashboard" class="hidden">

    <h1>구독 관리 대시보드</h1>
    <p class="muted">에코 AI 회원 및 입금 신청을 관리합니다.</p>

    <div class="cards">
        <div class="stat">
            <small>전체 조회 신청</small>
            <strong id="totalCount">0</strong>
        </div>

        <div class="stat">
            <small>승인 대기</small>
            <strong id="pendingCount">0</strong>
        </div>

        <div class="stat">
            <small>승인 완료</small>
            <strong id="approvedCount">0</strong>
        </div>
    </div>

    <section class="panel">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
            <h2>입금 신청 목록</h2>
            <button class="secondary" onclick="loadRequests()">새로고침</button>
        </div>

        <label>신청 상태 필터</label>

        <select id="statusFilter" onchange="loadRequests()">
            <option value="pending">승인 대기</option>
            <option value="approved">승인 완료</option>
            <option value="rejected">거절</option>
            <option value="all">전체 신청</option>
        </select>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>신청 번호</th>
                        <th>회원 아이디</th>
                        <th>입금자명</th>
                        <th>금액</th>
                        <th>신청 일시</th>
                        <th>상태</th>
                        <th>관리</th>
                    </tr>
                </thead>
                <tbody id="requestRows">
                    <tr><td colspan="7">신청 내역을 불러오는 중...</td></tr>
                </tbody>
            </table>
        </div>

        <div id="message"></div>
    </section>

    <button class="secondary" onclick="logout()">관리자 로그아웃</button>

</section>

</main>

<script>
let adminKey = "";

const loginPanel = document.getElementById("loginPanel");
const dashboard = document.getElementById("dashboard");

function showMessage(message) {
    document.getElementById("message").textContent = message;
}

async function adminRequest(path, method = "GET") {

    const response = await fetch(path, {
        method: method,
        headers: {
            "X-Admin-Key": adminKey,
            "Accept": "application/json"
        }
    });

    let data = {};

    try {
        data = await response.json();
    } catch (e) {
        throw new Error("서버 응답을 읽을 수 없습니다.");
    }

    if (!response.ok) {
        throw new Error(
            data.detail || "요청에 실패했습니다. HTTP " + response.status
        );
    }

    return data;
}

async function login() {

    const key = document.getElementById("adminKey").value.trim();

    if (!key) {
        document.getElementById("loginMessage").textContent =
            "관리자 비밀키를 입력해 주세요.";
        return;
    }

    adminKey = key;

    try {

        await adminRequest(
            "/admin/subscription-requests?status=pending"
        );

        loginPanel.classList.add("hidden");
        dashboard.classList.remove("hidden");

        await loadRequests();

    } catch (error) {

        adminKey = "";

        document.getElementById("loginMessage").textContent =
            error.message;
    }
}

function logout() {

    adminKey = "";

    document.getElementById("adminKey").value = "";

    dashboard.classList.add("hidden");
    loginPanel.classList.remove("hidden");

    document.getElementById("loginMessage").textContent =
        "로그아웃되었습니다.";
}

function formatDate(value) {

    if (!value) return "-";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) return value;

    return date.toLocaleString("ko-KR");
}

function statusLabel(status) {

    if (status === "approved") return "승인 완료";
    if (status === "rejected") return "거절";
    return "승인 대기";
}

function escapeText(value) {

    return String(value ?? "").replace(/[&<>"']/g, function(char) {
        return {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        }[char];
    });
}

async function loadRequests() {

    showMessage("신청 내역을 불러오는 중...");

    const status = document.getElementById("statusFilter").value;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests?status=" +
            encodeURIComponent(status)
        );

        const requests = data.requests || [];

        document.getElementById("totalCount").textContent =
            requests.length;

        document.getElementById("pendingCount").textContent =
            requests.filter(item => item.status === "pending").length;

        document.getElementById("approvedCount").textContent =
            requests.filter(item => item.status === "approved").length;

        const tbody = document.getElementById("requestRows");

        tbody.replaceChildren();

        if (requests.length === 0) {

            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 7;
            cell.textContent = "표시할 신청 내역이 없습니다.";

            row.appendChild(cell);
            tbody.appendChild(row);

        } else {

            for (const item of requests) {

                const row = document.createElement("tr");

                const values = [
                    "#" + item.request_id,
                    item.username,
                    item.depositor_name,
                    Number(item.amount).toLocaleString("ko-KR") + "원",
                    formatDate(item.created_at),
                    statusLabel(item.status)
                ];

                for (let index = 0; index < values.length; index++) {

                    const cell = document.createElement("td");

                    if (index === 5) {

                        const badge = document.createElement("span");

                        badge.className = "status " + item.status;
                        badge.textContent = values[index];

                        cell.appendChild(badge);

                    } else {

                        cell.textContent = values[index];
                    }

                    row.appendChild(cell);
                }

                const actionCell = document.createElement("td");

                if (item.status === "pending") {

                    const actions = document.createElement("div");
                    actions.className = "actions";

                    const approveButton = document.createElement("button");
                    approveButton.className = "approve";
                    approveButton.textContent = "승인";
                    approveButton.addEventListener("click", () => {
                        approveRequest(item.request_id, item.username);
                    });

                    const rejectButton = document.createElement("button");
                    rejectButton.className = "reject";
                    rejectButton.textContent = "거절";
                    rejectButton.addEventListener("click", () => {
                        rejectRequest(item.request_id, item.username);
                    });

                    actions.appendChild(approveButton);
                    actions.appendChild(rejectButton);
                    actionCell.appendChild(actions);

                } else {

                    actionCell.textContent = "-";
                }

                row.appendChild(actionCell);
                tbody.appendChild(row);
            }
        }

        showMessage("신청 내역을 업데이트했어.");

    } catch (error) {

        showMessage(error.message);
    }
}

async function approveRequest(id, username) {

    const confirmed = confirm(
        "회원 아이디: " + username +
        "\n신청 번호: " + id +
        "\n\n실제 입금 내역을 확인했어? 승인할까?"
    );

    if (!confirmed) return;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests/" + id + "/approve",
            "POST"
        );

        showMessage(
            data.message + "\n회원: " + data.username +
            "\n만료일: " + formatDate(data.expires_at)
        );

        await loadRequests();

    } catch (error) {

        showMessage(error.message);
    }
}

async function rejectRequest(id, username) {

    const confirmed = confirm(
        "회원 아이디: " + username +
        "\n신청 번호: " + id +
        "\n\n이 신청을 거절할까?"
    );

    if (!confirmed) return;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests/" + id + "/reject",
            "POST"
        );

        showMessage(data.message);

        await loadRequests();

    } catch (error) {

        showMessage(error.message);
    }
}

document.getElementById("adminKey").addEventListener(
    "keydown",
    function(event) {
        if (event.key === "Enter") login();
    }
);
</script>

</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin_homepage():
    return HTMLResponse(
        content=ADMIN_HTML,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# =====================================
# 18. AI 채팅
# =====================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    require_active_subscription(user, db)

    try:

        client = OpenAI(api_key=OPENAI_API_KEY)

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
