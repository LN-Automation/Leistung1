"""LN Automation – Dokumenten-Agent (Alles-in-einer-Datei-Version).

Start mit einem einzigen Befehl:
    streamlit run ln_app.py

Zwei Datenquellen:
  1) Dateien direkt im Browser hochladen (funktioniert sofort, nur API-Key nötig)
  2) Google-Cloud-Storage-Bucket verbinden (optional)

Die KI (Claude) bekommt pro Frage NUR die passendsten Dokument-Auszüge
aus der lokalen Vektordatenbank – sie kann also nichts anderes verwenden.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os


def setting(name: str, default: str = "") -> str:
    """Liest Einstellungen aus Streamlit-Secrets (Cloud) oder .env/Umgebung (lokal)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

# ----------------------------- Grund-Setup ---------------------------------

# Favicon: das Logo, falls es im Ordner liegt – sonst als Rückfall das Emoji
_LOGO_PFAD = Path(__file__).parent / "logo.png"
st.set_page_config(
    page_title="LN Automation – Dokumentensuche",
    page_icon=str(_LOGO_PFAD) if _LOGO_PFAD.exists() else "🤖",
    layout="centered",
)

# Streamlit-Menü, Deploy-Button und Footer ausblenden (cleaner Look für Kunden)
st.markdown(
    """
    <style>
      /* Menü und Deploy-Button weg – aber NICHT der Header: dort sitzt der
         Aufklapp-Pfeil der Sidebar. */
      #MainMenu, footer, .stAppDeployButton, [data-testid="stStatusWidget"] {visibility: hidden;}
      [data-testid="stHeader"] {background: transparent;}
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important;
        opacity: 1 !important; pointer-events: auto !important; z-index: 99999 !important;
      }
      .block-container {padding-top: 2.2rem;}
      .ln-section {
        font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: #64748b; font-weight: 600; margin: 2.2rem 0 0.6rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _logo_b64() -> str:
    """Logo als Base64 – damit es sich in HTML zentrieren lässt."""
    import base64

    if _LOGO_PFAD.exists():
        return base64.b64encode(_LOGO_PFAD.read_bytes()).decode()
    return ""


_logo = _logo_b64()


def _logo_block(breite: int = 230, untertitel: str = "") -> str:
    """HTML-Kopf mit zentriertem Logo. Ohne Logo-Datei: Schriftzug."""
    inneres = (
        f'<img src="data:image/png;base64,{_logo}" style="width:{breite}px;max-width:70%;" />'
        if _logo else
        '<div style="color:#0f172a;font-size:2rem;font-weight:800;">LN Automation</div>'
    )
    unter = (f'<div style="color:#64748b;font-size:1.0rem;margin-top:4px;">{untertitel}</div>'
             if untertitel else "")
    return (f'<div style="text-align:center;padding:10px 0 6px 0;margin-bottom:10px;">'
            f'{inneres}{unter}</div>')


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "kunde"


_kunden: dict[str, str] = {}
try:
    if "kunden" in st.secrets:
        _kunden = {str(k).lower(): str(v) for k, v in dict(st.secrets["kunden"]).items()}
except Exception:
    pass
_app_pw = setting("APP_PASSWORD")

if (_kunden or _app_pw) and not st.session_state.get("auth_ok"):
    st.markdown(
        _logo_block(210, "Enterprise KI-Dokumenten- und Datensuche für den Mittelstand")
        + '<hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 1.6rem 0;" />',
        unsafe_allow_html=True,
    )
    with st.form("login"):
        st.markdown("**Anmeldung**")
        firma = st.text_input("Firmen-Kennung") if _kunden else ""
        pw = st.text_input("Zugangspasswort", type="password")
        if st.form_submit_button("Anmelden", type="primary", width="stretch"):
            if _kunden:
                f = firma.strip().lower()
                if f in _kunden and pw == _kunden[f]:
                    st.session_state.auth_ok = True
                    st.session_state.kunde = firma.strip()
                    st.session_state.collection = "kunde_" + _slug(firma)
                    st.rerun()
                else:
                    st.error("Firmen-Kennung oder Passwort falsch.")
            elif pw == _app_pw:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()


# Kopfbereich – gleicher Baustein wie auf der Anmeldeseite
st.markdown(
    _logo_block(230, "Enterprise KI-Dokumenten- und Datensuche für den Mittelstand")
    + '<hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 1.4rem 0;" />',
    unsafe_allow_html=True,
)

COLLECTION = "dokumente"
EMB_MODEL = "intfloat/multilingual-e5-base"
EMB_DIM = 768
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300
TOP_K = 8
VOYAGE_DIM = 1024  # voyage-3.5-lite
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def use_voyage() -> bool:
    """Voyage-API aktiv, sobald ein VOYAGE_API_KEY hinterlegt ist (Secrets/.env)."""
    return bool(setting("VOYAGE_API_KEY"))


def emb_dim() -> int:
    return VOYAGE_DIM if use_voyage() else EMB_DIM


def current_collection() -> str:
    """Jeder angemeldete Kunde bekommt seine eigene, getrennte Collection.
    Suffix _v trennt Voyage-Indexe von lokalen (andere Vektorgröße)."""
    base = st.session_state.get("collection", COLLECTION)
    return base + ("_v" if use_voyage() else "")


# ------------------------- Bausteine (gecacht) ------------------------------

@st.cache_resource(show_spinner="Lade Suchmodell (nur beim ersten Start, ~1 GB) ...")
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMB_MODEL)


def _voyage_embed(texts: list[str], input_type: str) -> list[list[float]]:
    import requests

    import time

    out: list[list[float]] = []
    for i in range(0, len(texts), 12):  # kleine Pakete: passt auch ins freie Voyage-Limit
        batch = texts[i : i + 12]
        for _versuch in range(10):
            r = requests.post(
                VOYAGE_URL,
                headers={"Authorization": f"Bearer {setting('VOYAGE_API_KEY')}"},
                json={
                    "input": batch,
                    "model": setting("VOYAGE_MODEL", "voyage-3.5-lite"),
                    "input_type": input_type,
                },
                timeout=120,
            )
            if r.status_code == 429:
                time.sleep(22)  # Voyage bremst (Free-Limit) -> warten und erneut
                continue
            break
        if r.status_code != 200:
            try:
                j = r.json()
                detail = j.get("detail") or j.get("error", {}).get("message") or str(j)[:300]
            except Exception:  # noqa: BLE001
                detail = r.text[:300]
            hint = {
                401: "Der VOYAGE_API_KEY in den Secrets ist falsch oder unvollständig.",
                402: "Voyage-Guthaben/Zahlungsmethode fehlt – im Voyage-Dashboard unter Billing prüfen.",
                429: "Rate-Limit erreicht – kurz warten und erneut versuchen.",
            }.get(r.status_code, "")
            raise RuntimeError(f"Voyage-API-Fehler {r.status_code}: {detail} {hint}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    return out


def embed_passages(texts: list[str]) -> list[list[float]]:
    if use_voyage():
        return _voyage_embed(texts, "document")
    return get_embedder().encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True
    ).tolist()


def embed_query(text: str) -> list[float]:
    if use_voyage():
        return _voyage_embed([text], "query")[0]
    return get_embedder().encode(f"query: {text}", normalize_embeddings=True).tolist()


@st.cache_resource
def get_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    url = setting("QDRANT_URL")
    if url:
        client = QdrantClient(url=url, api_key=setting("QDRANT_API_KEY") or None)
    else:
        client = QdrantClient(path="./qdrant_data")
    return client


def qdrant():
    """Liefert den Client und stellt sicher, dass die Collection des Kunden existiert."""
    from qdrant_client.models import Distance, VectorParams

    base = get_qdrant()
    name = current_collection()
    if not base.collection_exists(name):
        base.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=emb_dim(), distance=Distance.COSINE),
        )
    # Qdrant Cloud verlangt fuer Filter (z.B. Loeschen nach Dateiname) ein
    # Payload-Register auf dem Feld "source" - anlegen ist idempotent genug:
    try:
        from qdrant_client.models import PayloadSchemaType

        base.create_payload_index(
            collection_name=name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:  # noqa: BLE001
        pass  # existiert bereits
    return base


# --------------------------- Text-Extraktion --------------------------------

def _shrink_image(data: bytes) -> tuple[bytes, str]:
    """Verkleinert Bilder auf API-taugliche Größe, gibt (bytes, media_type) zurück."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.thumbnail((2000, 2000))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001
        return data, "image/jpeg"


def ocr_with_claude(image_bytes: bytes, media_type: str, api_key: str) -> str:
    """Liest Text aus einem Bild/Scan über die Claude-API (Vision)."""
    import base64

    from anthropic import Anthropic

    msg = Anthropic(api_key=api_key).messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode(),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transkribiere den gesamten sichtbaren Text dieses Dokuments "
                            "exakt und vollständig (inkl. Zahlen, Beträge, Daten, Tabellen "
                            "als Text). Gib NUR den transkribierten Inhalt aus, keine "
                            "Kommentare."
                        ),
                    },
                ],
            }
        ],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def extract_text(filename: str, data: bytes, api_key: str = "") -> list[tuple[str, str]]:
    """Liefert Liste von (seiten_label, text)."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in (".png", ".jpg", ".jpeg", ".webp"):
            if not api_key:
                st.warning(f"{filename}: Für Bilder wird der API-Key benötigt (links eintragen).")
                return []
            small, mtype = _shrink_image(data)
            t = ocr_with_claude(small, mtype, api_key)
            return [("Bild (KI-Texterkennung)", t)] if t else []
        if suffix == ".pdf":
            import fitz

            pages = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for i, page in enumerate(doc, start=1):
                    t = page.get_text("text").strip()
                    if not t and api_key:
                        # Gescannte Seite ohne Textebene -> als Bild an Claude
                        pix = page.get_pixmap(dpi=150)
                        t = ocr_with_claude(pix.tobytes("jpeg"), "image/jpeg", api_key)
                        if t:
                            pages.append((f"Seite {i} (Scan, KI-Texterkennung)", t))
                        continue
                    if t:
                        pages.append((f"Seite {i}", t))
            return pages
        if suffix == ".docx":
            from docx import Document

            d = Document(io.BytesIO(data))
            t = "\n".join(p.text for p in d.paragraphs if p.text.strip())
            return [("Dokument", t)] if t else []
        if suffix in (".xlsx", ".xls"):
            import pandas as pd

            engine = "xlrd" if suffix == ".xls" else "openpyxl"
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine)
            return [
                (f"Blatt '{name}'", df.to_markdown(index=False))
                for name, df in sheets.items()
                if not df.empty
            ]
        if suffix in (".txt", ".md", ".csv"):
            return [("Dokument", data.decode("utf-8", errors="replace"))]
    except Exception as e:  # noqa: BLE001
        st.warning(f"{filename}: konnte nicht gelesen werden ({e})")
    return []


def chunk(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return out


# ----------------------------- Indexierung ----------------------------------

def index_document(filename: str, data: bytes, quelle: str, api_key: str = "") -> int:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
    )

    client = qdrant()

    # Alte Version dieses Dokuments entfernen
    client.delete(
        collection_name=current_collection(),
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]),
    )

    chunks: list[dict] = []
    for page_label, text in extract_text(filename, data, api_key):
        for piece in chunk(text):
            chunks.append({"source": filename, "page": page_label, "text": piece, "herkunft": quelle})
    if not chunks:
        return 0

    vectors = embed_passages([c["text"] for c in chunks])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
        for c, v in zip(chunks, vectors)
    ]
    # In Päckchen speichern - Qdrant Cloud erlaubt max. 32 MB pro Sendung
    for i in range(0, len(points), 64):
        client.upsert(
            collection_name=current_collection(),
            points=points[i : i + 64],
        )
    return len(chunks)


def indexed_files() -> dict[str, int]:
    """Alle indexierten Dokumente mit Anzahl ihrer Abschnitte."""
    client = qdrant()
    counts: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=current_collection(), limit=500, with_payload=True, offset=offset
        )
        for p in points:
            src = p.payload.get("source", "?")
            counts[src] = counts.get(src, 0) + 1
        if offset is None:
            break
    return dict(sorted(counts.items()))


def delete_source(source: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant().delete(
        collection_name=current_collection(),
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
    )


def clear_all() -> None:
    from qdrant_client.models import Distance, VectorParams

    c = qdrant()
    c.delete_collection(current_collection())
    c.create_collection(
        collection_name=current_collection(),
        vectors_config=VectorParams(size=emb_dim(), distance=Distance.COSINE),
    )


# ------------------------------- Suche + KI ---------------------------------

def ask_claude(question: str, api_key: str) -> tuple[str, list[dict]]:
    from anthropic import Anthropic

    client = qdrant()

    qvec = embed_query(question)
    hits = client.query_points(
        collection_name=current_collection(), query=qvec, limit=TOP_K, with_payload=True
    ).points
    if not hits:
        return "Es sind noch keine Dokumente indexiert.", []

    context = "\n\n".join(
        f'<auszug quelle="{h.payload["source"]}" seite="{h.payload["page"]}">\n{h.payload["text"]}\n</auszug>'
        for h in hits
    )
    system = (
        "Du bist der Dokumentenassistent von LN Automation. Beantworte Fragen "
        "AUSSCHLIESSLICH auf Basis der bereitgestellten Dokumentauszüge. Gib bei jeder "
        "Aussage die Quelle an im Format [Dateiname, Seite]. Zahlen und Beträge exakt "
        "wiedergeben. Wenn die Auszüge die Frage nicht beantworten, sage das klar und "
        "rate nicht. Antworte auf Deutsch. Formatierung: normaler Fließtext, bei "
        "Aufzählungen einfache Spiegelstriche; verwende NIEMALS Markdown-Überschriften "
        "(#, ##) und kein übermäßiges Fettdruck-Formatieren."
    )
    msg = Anthropic(api_key=api_key).messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": f"Dokumentauszüge:\n\n{context}\n\nFrage: {question}"}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")

    sources, seen = [], set()
    for h in hits:
        key = (h.payload["source"], h.payload["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1], "score": round(h.score, 3)})
    return answer, sources


# --------------------------------- Sidebar ----------------------------------

with st.sidebar:
    st.header("Einrichtung")
    if st.session_state.get("kunde"):
        st.caption(f"Angemeldet: {st.session_state.kunde}")
        if st.button("Abmelden"):
            for _k in ("auth_ok", "kunde", "collection"):
                st.session_state.pop(_k, None)
            st.rerun()
        st.divider()
    if "api_key" not in st.session_state:
        st.session_state.api_key = setting("ANTHROPIC_API_KEY")

    if st.session_state.api_key:
        st.success("API-Key aktiv ✓")
        if st.button("Key ändern"):
            st.session_state.api_key = ""
            st.rerun()
    else:
        with st.form("key_form"):
            _k = st.text_input(
                "Claude API-Key",
                type="password",
                help="Von console.anthropic.com – beginnt mit sk-ant-",
            )
            if st.form_submit_button("Speichern", type="primary"):
                _k = _k.strip()
                if _k.startswith("sk-ant-") and len(_k) > 20:
                    st.session_state.api_key = _k
                    st.rerun()
                else:
                    st.error("Das sieht nicht wie ein Claude-Key aus (er beginnt mit sk-ant-).")

    api_key = st.session_state.api_key

# ------------------------------ Datenquellen --------------------------------

st.markdown('<div class="ln-section">Dokumente verbinden</div>', unsafe_allow_html=True)
tab_upload, tab_cloud = st.tabs(["Dateien hochladen", "Cloud verbinden"])

with tab_upload:
    uploads = st.file_uploader(
        "PDF, Word, Excel, CSV, Text oder Bilder/Scans (PNG, JPG) – mehrere gleichzeitig möglich",
        type=["pdf", "docx", "xlsx", "xls", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Hochgeladene Dateien indexieren", type="primary"):
        try:
            total = 0
            prog = st.progress(0.0)
            for i, up in enumerate(uploads, start=1):
                n = index_document(up.name, up.getvalue(), quelle="Upload", api_key=api_key)
                total += n
                prog.progress(i / len(uploads), text=f"{up.name}: {n} Abschnitte")
            st.success(f"Fertig – {total} Abschnitte aus {len(uploads)} Dateien indexiert.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Indexierung fehlgeschlagen: {e}")

def _drive_service():
    """Google-Drive-Zugriff über den Service Account (Secrets oder lokale JSON)."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]), scopes=scopes
            )
    except Exception:  # noqa: BLE001
        pass
    if creds is None:
        keyfile = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service-account.json")
        creds = service_account.Credentials.from_service_account_file(keyfile, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _drive_folder_id(link_or_id: str) -> str:
    import re

    m = re.search(r"/folders/([A-Za-z0-9_-]+)", link_or_id)
    return m.group(1) if m else link_or_id.strip()


def _drive_walk(svc, folder_id: str, files: list, limit: int = 300) -> None:
    """Sammelt rekursiv alle Dateien eines Drive-Ordners (inkl. Unterordner)."""
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                _drive_walk(svc, f["id"], files, limit)
            else:
                files.append(f)
            if len(files) >= limit:
                return
        page_token = resp.get("nextPageToken")
        if not page_token:
            return


def _drive_download(svc, f: dict) -> tuple[str, bytes] | None:
    """Lädt eine Drive-Datei; Google-Formate werden passend exportiert."""
    mime = f["mimeType"]
    if mime == "application/vnd.google-apps.document":
        data = svc.files().export(fileId=f["id"], mimeType="text/plain").execute()
        return f["name"] + ".txt", data
    if mime == "application/vnd.google-apps.spreadsheet":
        data = svc.files().export(
            fileId=f["id"],
            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).execute()
        return f["name"] + ".xlsx", data
    if mime.startswith("application/vnd.google-apps"):
        return None  # andere Google-Formate (Slides etc.) vorerst überspringen
    return f["name"], svc.files().get_media(fileId=f["id"]).execute()


with tab_cloud:
    anbieter = st.radio(
        "Cloud-Anbieter",
        ["Google Drive", "Microsoft OneDrive"],
        horizontal=True,
    )
    if anbieter.startswith("Microsoft"):
        st.caption(
            "Schritt 1: In OneDrive Rechtsklick auf den Ordner -> Teilen -> "
            "Linkeinstellungen: 'Jeder, der über den Link verfügt' (Anzeigen) -> Link kopieren."
        )
        od_link = st.text_input("Schritt 2: OneDrive-Freigabelink einfügen")
        if od_link and st.button("OneDrive-Ordner verbinden und indexieren", type="primary"):
            import base64

            import requests as _rq

            def _od_share_id(url: str) -> str:
                b = base64.urlsafe_b64encode(url.strip().encode()).decode().rstrip("=")
                return "u!" + b

            def _od_json(url: str) -> dict:
                r = _rq.get(url, timeout=60)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"OneDrive-Fehler {r.status_code}. Ist der Link auf "
                        f"'Jeder, der über den Link verfügt' gestellt? ({r.text[:150]})"
                    )
                return r.json()

            def _od_children(share_id: str, path: str) -> list[dict]:
                base = f"https://api.onedrive.com/v1.0/shares/{share_id}/driveItem"
                url = base + (f":/{path}:/children" if path else "/children")
                items: list[dict] = []
                while url:
                    data = _od_json(url)
                    items.extend(data.get("value", []))
                    url = data.get("@odata.nextLink")
                return items

            def _od_walk(share_id: str, path: str, files: list, limit: int = 300) -> None:
                for it in _od_children(share_id, path):
                    if "folder" in it:
                        _od_walk(
                            share_id,
                            (path + "/" if path else "") + it["name"],
                            files,
                            limit,
                        )
                    else:
                        files.append(it)
                    if len(files) >= limit:
                        return

            try:
                share_id = _od_share_id(od_link)
                files: list = []
                with st.spinner("Lese OneDrive-Ordner ..."):
                    _od_walk(share_id, "", files)
                if not files:
                    st.warning("Keine Dateien gefunden - Link und Freigabe prüfen.")
                else:
                    total = 0
                    prog = st.progress(0.0)
                    for i, f in enumerate(files, start=1):
                        dl = f.get("@microsoft.graph.downloadUrl") or f.get(
                            "@content.downloadUrl"
                        )
                        if not dl:
                            continue
                        data = _rq.get(dl, timeout=120).content
                        n = index_document(
                            f["name"], data, quelle="OneDrive", api_key=api_key
                        )
                        total += n
                        prog.progress(i / len(files), text=f"{f['name']}: {n} Abschnitte")
                    st.success(
                        f"Fertig - {total} Abschnitte aus {len(files)} OneDrive-Dateien indexiert."
                    )
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"OneDrive-Verbindung fehlgeschlagen: {e}")
    else:
        sa_email = ""
        try:
            sa_email = dict(st.secrets.get("gcp_service_account", {})).get("client_email", "")
        except Exception:  # noqa: BLE001
            pass
        if sa_email:
            st.caption(f"Schritt 1: Drive-Ordner freigeben für **{sa_email}** (als Betrachter).")
        else:
            st.caption(
                "Schritt 1: In den Secrets den [gcp_service_account]-Block hinterlegen, "
                "dann den Drive-Ordner für dessen E-Mail-Adresse freigeben."
            )
        drive_link = st.text_input(
            "Schritt 2: Link des Drive-Ordners einfügen",
            help="In Google Drive: Rechtsklick auf den Ordner -> Link abrufen -> hier einfügen.",
        )
        if drive_link and st.button("Drive-Ordner verbinden und indexieren", type="primary"):
            try:
                svc = _drive_service()
                files: list = []
                with st.spinner("Lese Ordnerinhalt ..."):
                    _drive_walk(svc, _drive_folder_id(drive_link), files)
                if not files:
                    st.warning(
                        "Keine Dateien gefunden. Ist der Ordner für den Service Account "
                        "freigegeben und der Link korrekt?"
                    )
                else:
                    total = 0
                    prog = st.progress(0.0)
                    for i, f in enumerate(files, start=1):
                        loaded = _drive_download(svc, f)
                        if loaded is None:
                            continue
                        name, data = loaded
                        n = index_document(name, data, quelle="Drive", api_key=api_key)
                        total += n
                        prog.progress(i / len(files), text=f"{name}: {n} Abschnitte")
                    st.success(
                        f"Fertig – {total} Abschnitte aus {len(files)} Drive-Dateien indexiert."
                    )
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Drive-Verbindung fehlgeschlagen: {e}")

# ------------------------ Indexierte Dokumente ------------------------------

st.markdown('<div class="ln-section">Indexierte Dokumente</div>', unsafe_allow_html=True)

docs = indexed_files()
if not docs:
    st.caption("Noch keine Dokumente im Index – oben hochladen und indexieren.")
else:
    st.caption(f"{len(docs)} Dokument(e), {sum(docs.values())} durchsuchbare Abschnitte")
    for src, n in docs.items():
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"📄 **{src}**  \n<span style='color:#94a3b8;font-size:0.85rem;'>{n} Abschnitte</span>", unsafe_allow_html=True)
        if c2.button("Entfernen", key=f"del_{src}"):
            delete_source(src)
            st.rerun()

    if st.session_state.get("confirm_clear"):
        st.warning("Wirklich ALLE Dokumente aus dem Index entfernen?")
        cc1, cc2 = st.columns([1, 1])
        if cc1.button("Ja, alle entfernen", type="primary"):
            clear_all()
            st.session_state.confirm_clear = False
            st.rerun()
        if cc2.button("Abbrechen"):
            st.session_state.confirm_clear = False
            st.rerun()
    else:
        if st.button("🧹 Alle entfernen"):
            st.session_state.confirm_clear = True
            st.rerun()

# --------------------------------- Chat -------------------------------------

st.markdown('<div class="ln-section">Fragen stellen</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("z.B. Was haben wir laut Rechnung X am Tag Y für Produkt Z gezahlt?")
if question:
    if not api_key:
        st.error("Bitte zuerst links den Claude API-Key eintragen.")
        st.stop()
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Durchsuche Dokumente ..."):
            try:
                answer, sources = ask_claude(question, api_key)
                st.markdown(answer)
                if sources:
                    with st.expander("Verwendete Quellen"):
                        for s in sources:
                            st.markdown(f"- **{s['source']}** ({s['page']}, Relevanz {s['score']})")
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:  # noqa: BLE001
                st.error(f"Fehler: {e}")
