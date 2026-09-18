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

from datetime import datetime

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
      /* Menü, Deploy-Button und die Cloud-Leiste ("Fork", GitHub) ausblenden.
         Der Header selbst bleibt bestehen, weil dort der Aufklapp-Pfeil der
         Sidebar sitzt – dieser wird darunter gezielt wieder sichtbar gemacht.
         visibility: visible beim Kind schlaegt visibility: hidden beim Eltern-Element. */
      #MainMenu, footer, .stAppDeployButton, [data-testid="stStatusWidget"],
      [data-testid="stToolbar"], [data-testid="stToolbarActions"],
      [data-testid="stAppToolbar"], .stAppToolbar {visibility: hidden;}
      [data-testid="stHeader"] {background: transparent;}
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapsedControl"] *,
      [data-testid="collapsedControl"],
      [data-testid="collapsedControl"] * {
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
    # Linie bewusst nur unter dem Schriftzug statt über die ganze Breite
    return (f'<div style="text-align:center;padding:26px 0 10px 0;">'
            f'{inneres}{unter}'
            f'<hr style="border:none;border-top:1px solid #e2e8f0;'
            f'width:min(560px,60%);margin:22px auto 0 auto;" />'
            f'</div>')


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
        _logo_block(210, "Enterprise KI-Dokumenten- und Datensuche für den Mittelstand"),
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
    + '<div style="height:26px;"></div>',
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

def inhalt_hash(data: bytes) -> str:
    """Fingerabdruck des Dateiinhalts – erkennt dieselbe Datei unter
    anderem Namen, etwa „Vertrag.pdf" und „Vertrag (1).pdf"."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


def dublette(data: bytes, dateiname: str) -> str | None:
    """Liefert den Namen des bereits indexierten Dokuments mit gleichem
    Inhalt, sonst None."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        client = qdrant()
        pts, _ = client.scroll(
            collection_name=current_collection(), limit=1, with_payload=True,
            scroll_filter=Filter(must=[FieldCondition(
                key="hash", match=MatchValue(value=inhalt_hash(data)))]),
        )
        for p_ in pts:
            vorhanden = p_.payload.get("source")
            if vorhanden and vorhanden != dateiname:
                return vorhanden
    except Exception:  # noqa: BLE001
        pass
    return None


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
            chunks.append({"source": filename, "page": page_label,
                           "text": piece, "herkunft": quelle,
                           "hash": inhalt_hash(data)})
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

MODELL = "claude-sonnet-4-6"
MODELL_KLEIN = "claude-haiku-4-5"      # nur zum Umformulieren von Folgefragen
MIN_SCORE = 0.30                       # darunter gilt ein Treffer als unbrauchbar
REL_ANTEIL = 0.55                      # Treffer unter 55 % des besten fallen raus


def suchanfrage(frage: str, verlauf: list[dict], api_key: str) -> str:
    """Folgefragen in eine eigenständige Suchanfrage umschreiben.

    „Und was zur Haftung?" findet für sich genommen nichts. Ohne diesen
    Schritt sucht die Vektorsuche wörtlich nach dem Fragment und liefert
    die falschen Abschnitte.
    """
    if not verlauf:
        return frage
    letzte = verlauf[-6:]
    gespraech = "\n".join(
        f"{'Frage' if m['role'] == 'user' else 'Antwort'}: {m['content'][:400]}"
        for m in letzte
    )
    try:
        from anthropic import Anthropic

        msg = Anthropic(api_key=api_key).messages.create(
            model=MODELL_KLEIN,
            max_tokens=150,
            system=("Formuliere die letzte Frage so um, dass sie ohne den "
                    "Gesprächsverlauf verständlich ist. Nur die umformulierte "
                    "Frage ausgeben, nichts sonst. Ist die Frage bereits "
                    "eigenständig, gib sie unverändert zurück."),
            messages=[{"role": "user",
                       "content": f"Bisheriges Gespräch:\n{gespraech}\n\n"
                                  f"Letzte Frage: {frage}"}],
        )
        neu_text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return neu_text or frage
    except Exception:  # noqa: BLE001 – im Zweifel die Originalfrage
        return frage


def belegstellen(question: str, api_key: str, verlauf: list[dict] | None = None,
                 nur_quellen: list[str] | None = None) -> tuple[str, list[dict], str]:
    """Passende Abschnitte suchen.

    Liefert (Kontext für das Modell, Belegstellen, Hinweis). Ist der Hinweis
    gefüllt, gibt es keine brauchbaren Treffer und es wird gar nicht gefragt.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    verlauf = verlauf or []
    client = qdrant()

    gesucht = suchanfrage(question, verlauf, api_key)
    qvec = embed_query(gesucht)
    bedingung = None
    if nur_quellen:
        bedingung = Filter(must=[FieldCondition(key="source",
                                                match=MatchAny(any=nur_quellen))])
    hits = client.query_points(
        collection_name=current_collection(), query=qvec, limit=TOP_K,
        with_payload=True, query_filter=bedingung,
    ).points
    if not hits:
        return "", [], "Es sind noch keine passenden Dokumente indexiert."

    # Schwelle: schwache Treffer fliegen raus, statt dass die KI aus
    # unpassenden Auszügen etwas zusammenreimt.
    bester = max(h.score for h in hits)
    hits = [h for h in hits
            if h.score >= MIN_SCORE and h.score >= bester * REL_ANTEIL]
    if not hits:
        return "", [], (
            "Dazu steht nichts in den indexierten Dokumenten. Die Suche hat "
            "keinen ausreichend passenden Abschnitt gefunden – bitte die Frage "
            "anders formulieren oder das passende Dokument ergänzen."
        )

    context = "\n\n".join(
        f'<auszug quelle="{h.payload["source"]}" seite="{h.payload["page"]}">\n'
        f'{h.payload["text"]}\n</auszug>'
        for h in hits
    )
    quellen, gesehen = [], set()
    for h in hits:
        key = (h.payload["source"], h.payload["page"])
        if key not in gesehen:
            gesehen.add(key)
            quellen.append({
                "source": key[0], "page": key[1], "score": round(h.score, 3),
                "text": (h.payload.get("text") or "")[:600],
            })
    return context, quellen, ""


SYSTEM_PROMPT = (
    "Du bist der Dokumentenassistent von LN Automation. Beantworte Fragen "
    "AUSSCHLIESSLICH auf Basis der bereitgestellten Dokumentauszüge. Gib bei jeder "
    "Aussage die Quelle an im Format [Dateiname, Seite]. Zahlen und Beträge exakt "
    "wiedergeben. Wenn die Auszüge die Frage nicht beantworten, sage das klar und "
    "rate nicht. Antworte auf Deutsch. Formatierung: normaler Fließtext, bei "
    "Aufzählungen einfache Spiegelstriche; verwende NIEMALS Markdown-Überschriften "
    "(#, ##) und kein übermäßiges Fettdruck-Formatieren."
)


def antwort_stream(question: str, context: str, verlauf: list[dict],
                   api_key: str, ergebnis: dict):
    """Antwort stückweise liefern, damit nicht 20 Sekunden nur ein Spinner
    läuft. Der Tokenverbrauch landet danach in `ergebnis`."""
    from anthropic import Anthropic

    nachrichten = [{"role": m["role"], "content": m["content"]}
                   for m in (verlauf or [])[-8:]]
    nachrichten.append({
        "role": "user",
        "content": f"Dokumentauszüge:\n\n{context}\n\nFrage: {question}",
    })
    with Anthropic(api_key=api_key).messages.stream(
        model=MODELL, max_tokens=1500, system=SYSTEM_PROMPT,
        messages=nachrichten,
    ) as strom:
        for stueck in strom.text_stream:
            yield stueck
        ergebnis["usage"] = getattr(strom.get_final_message(), "usage", None)


def ask_claude(question: str, api_key: str, verlauf: list[dict] | None = None,
               nur_quellen: list[str] | None = None) -> tuple[str, list[dict], dict]:
    """Ohne Streaming – bleibt für Aufrufe von außen erhalten."""
    from anthropic import Anthropic

    from ln_nutzung import datensatz

    context, quellen, hinweis = belegstellen(question, api_key, verlauf,
                                             nur_quellen)
    if hinweis:
        return hinweis, [], datensatz("frage", MODELL, None)

    nachrichten = [{"role": m["role"], "content": m["content"]}
                   for m in (verlauf or [])[-8:]]
    nachrichten.append({
        "role": "user",
        "content": f"Dokumentauszüge:\n\n{context}\n\nFrage: {question}",
    })
    msg = Anthropic(api_key=api_key).messages.create(
        model=MODELL, max_tokens=1500, system=SYSTEM_PROMPT,
        messages=nachrichten,
    )
    antwort = "".join(b.text for b in msg.content if b.type == "text")
    return antwort, quellen, datensatz("frage", MODELL,
                                       getattr(msg, "usage", None))


# ----------------------------- Unterhaltungen -------------------------------

def _coll_chats() -> str:
    return "chats_" + _slug(st.session_state.get("kunde", "lokal"))


def _coll_nutzung() -> str:
    return "nutzung_" + _slug(st.session_state.get("kunde", "lokal"))


def _sicherstellen(name: str):
    from qdrant_client.models import Distance, VectorParams

    c = qdrant()
    if not c.collection_exists(name):
        c.create_collection(collection_name=name,
                            vectors_config=VectorParams(size=1,
                                                        distance=Distance.COSINE))
    return c


def chats_laden() -> list[dict]:
    try:
        c = _sicherstellen(_coll_chats())
        pts, _ = c.scroll(collection_name=_coll_chats(), limit=100,
                          with_payload=True)
        chats = [p.payload for p in pts]
        return sorted(chats, key=lambda x: x.get("zuletzt", ""), reverse=True)
    except Exception:  # noqa: BLE001
        return st.session_state.get("_chats_lokal", [])


def chat_speichern(chat: dict) -> None:
    lokal = st.session_state.setdefault("_chats_lokal", [])
    st.session_state["_chats_lokal"] = [c for c in lokal
                                        if c.get("id") != chat["id"]] + [chat]
    try:
        from qdrant_client.models import PointStruct

        c = _sicherstellen(_coll_chats())
        c.upsert(collection_name=_coll_chats(),
                 points=[PointStruct(id=chat["id"], vector=[0.0], payload=chat)])
    except Exception:  # noqa: BLE001
        pass


def chat_loeschen(chat_id: str) -> None:
    st.session_state["_chats_lokal"] = [
        c for c in st.session_state.get("_chats_lokal", [])
        if c.get("id") != chat_id
    ]
    try:
        c = qdrant()
        c.delete(collection_name=_coll_chats(), points_selector=[chat_id])
    except Exception:  # noqa: BLE001
        pass


def nutzung_speichern(satz: dict) -> None:
    st.session_state.setdefault("_nutzung_lokal", []).append(satz)
    try:
        import uuid as _uuid

        from qdrant_client.models import PointStruct

        c = _sicherstellen(_coll_nutzung())
        c.upsert(collection_name=_coll_nutzung(),
                 points=[PointStruct(id=str(_uuid.uuid4()), vector=[0.0],
                                     payload=satz)])
    except Exception:  # noqa: BLE001
        pass


def nutzung_laden() -> list[dict]:
    try:
        c = _sicherstellen(_coll_nutzung())
        out, offset = [], None
        while True:
            pts, offset = c.scroll(collection_name=_coll_nutzung(), limit=500,
                                   with_payload=True, offset=offset)
            out.extend(p.payload for p in pts)
            if offset is None:
                return out
    except Exception:  # noqa: BLE001
        return st.session_state.get("_nutzung_lokal", [])


def einstellungen_laden() -> dict:
    try:
        c = _sicherstellen("einstellungen_" + _slug(st.session_state.get("kunde", "lokal")))
        pts = c.retrieve(
            collection_name="einstellungen_" + _slug(st.session_state.get("kunde", "lokal")),
            ids=[1], with_payload=True)
        return dict(pts[0].payload) if pts else {}
    except Exception:  # noqa: BLE001
        return st.session_state.get("_einst_lokal", {})


def einstellungen_speichern(daten: dict) -> None:
    st.session_state["_einst_lokal"] = daten
    try:
        from qdrant_client.models import PointStruct

        name = "einstellungen_" + _slug(st.session_state.get("kunde", "lokal"))
        c = _sicherstellen(name)
        c.upsert(collection_name=name,
                 points=[PointStruct(id=1, vector=[0.0], payload=daten)])
    except Exception:  # noqa: BLE001
        pass


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

    # ---- Unterhaltungen ------------------------------------------------
    st.divider()
    st.markdown('<div class="ln-section">Unterhaltungen</div>',
                unsafe_allow_html=True)

    if st.button("Neue Unterhaltung", type="primary", width="stretch"):
        st.session_state["chat_id"] = None
        st.session_state["history"] = []
        st.rerun()

    for _c in chats_laden()[:25]:
        z1, z2 = st.columns([5, 1])
        aktiv = _c.get("id") == st.session_state.get("chat_id")
        if z1.button(("● " if aktiv else "") + (_c.get("titel") or "Ohne Titel")[:38],
                     key=f"chat_{_c['id']}", width="stretch"):
            st.session_state["chat_id"] = _c["id"]
            st.session_state["history"] = _c.get("verlauf", [])
            st.rerun()
        if z2.button("✕", key=f"delchat_{_c['id']}", help="Unterhaltung löschen"):
            chat_loeschen(_c["id"])
            if aktiv:
                st.session_state["chat_id"] = None
                st.session_state["history"] = []
            st.rerun()

# --------------------------------- Reiter -----------------------------------

from ln_nutzung import datensatz, render_nutzung  # noqa: E402

tab_chat, tab_dok, tab_nutzung = st.tabs(["Chat", "Dokumente", "Nutzung"])

# ------------------------------ Datenquellen --------------------------------

with tab_dok:
    st.markdown('<div class="ln-section">Dokumente verbinden</div>', unsafe_allow_html=True)
    tab_upload, tab_cloud = st.tabs(["Dateien hochladen", "Cloud verbinden"])

    with tab_upload:
        _m = st.session_state.pop("index_meldung", None)
        if _m:
            if _m["neu"] or _m["ersetzt"]:
                st.success(
                    f"{len(_m['neu']) + len(_m['ersetzt'])} Datei(en) indexiert, "
                    f"{_m['total']} durchsuchbare Abschnitte."
                )
                for _n, _z in _m["neu"]:
                    st.caption(f"neu · {_n} · {_z} Abschnitte")
                for _n, _z in _m["ersetzt"]:
                    st.caption(f"aktualisiert · {_n} · {_z} Abschnitte")
            if _m["doppelt"]:
                st.info(
                    f"{len(_m['doppelt'])} Datei(en) übersprungen – gleicher "
                    f"Inhalt ist schon im Index: "
                    + ", ".join(f"{a} (wie {b})" for a, b in _m["doppelt"])
                )
            if _m["leer"]:
                st.warning(
                    "Kein Text gefunden in: " + ", ".join(_m["leer"])
                    + ". Bei Scans hilft ein Bild statt eines leeren PDFs."
                )

        uploads = st.file_uploader(
            "PDF, Word, Excel, CSV, Text oder Bilder/Scans (PNG, JPG) – mehrere gleichzeitig möglich",
            type=["pdf", "docx", "xlsx", "xls", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Hochgeladene Dateien indexieren",
                                 type="primary", width="stretch"):
            try:
                vorher = set(indexed_files())
                total, neu, ersetzt, doppelt, leer = 0, [], [], [], []
                prog = st.progress(0.0)
                for i, up in enumerate(uploads, start=1):
                    rohdaten = up.getvalue()
                    schon_da = dublette(rohdaten, up.name)
                    if schon_da:
                        doppelt.append((up.name, schon_da))
                        prog.progress(i / len(uploads),
                                      text=f"{up.name}: bereits vorhanden")
                        continue
                    n = index_document(up.name, rohdaten, quelle="Upload",
                                       api_key=api_key)
                    total += n
                    if n == 0:
                        leer.append(up.name)
                    elif up.name in vorher:
                        ersetzt.append((up.name, n))
                    else:
                        neu.append((up.name, n))
                    prog.progress(i / len(uploads),
                                  text=f"{up.name}: {n} Abschnitte")
                st.session_state["index_meldung"] = {
                    "total": total, "neu": neu, "ersetzt": ersetzt,
                    "doppelt": doppelt, "leer": leer,
                }
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

    st.markdown('<div class="ln-section">Indexierte Dokumente</div>',
                unsafe_allow_html=True)

    docs = indexed_files()
    if not docs:
        st.caption("Noch keine Dokumente im Index – oben hochladen und indexieren.")
    else:
        st.caption(f"{len(docs)} Dokument(e), {sum(docs.values())} "
                   f"durchsuchbare Abschnitte")

        # Bei vielen Dokumenten ist eine ungefilterte Liste unbrauchbar.
        such = st.text_input("Dokument suchen", placeholder="Name eingeben …",
                             key="dok_suche")
        treffer = {k: v for k, v in docs.items()
                   if such.lower() in k.lower()} if such else docs

        if st.session_state.get("confirm_clear"):
            st.warning("Wirklich ALLE Dokumente aus dem Index entfernen?")
            cc1, cc2 = st.columns(2)
            if cc1.button("Ja, alle entfernen", type="primary", width="stretch"):
                clear_all()
                st.session_state.confirm_clear = False
                st.rerun()
            if cc2.button("Abbrechen", width="stretch"):
                st.session_state.confirm_clear = False
                st.rerun()
        elif st.button(f"Alle {len(docs)} Dokumente entfernen"):
            st.session_state.confirm_clear = True
            st.rerun()

        with st.expander(f"Liste anzeigen ({len(treffer)} von {len(docs)})",
                         expanded=bool(such) or len(docs) <= 8):
            for src, n in sorted(treffer.items()):
                c1, c2 = st.columns([6, 1])
                c1.markdown(
                    f"📄 **{src}**  \n<span style='color:#94a3b8;"
                    f"font-size:0.85rem;'>{n} Abschnitte</span>",
                    unsafe_allow_html=True)
                if c2.button("Entfernen", key=f"del_{src}"):
                    delete_source(src)
                    st.rerun()
            if not treffer:
                st.caption("Kein Dokument mit diesem Namen.")

# --------------------------------- Chat -------------------------------------

with tab_chat:
    docs_alle = indexed_files()

    if not docs_alle:
        st.info("Noch keine Dokumente im Index. Im Reiter „Dokumente“ "
                "hochladen oder eine Cloud verbinden.")
    else:
        k1, k2 = st.columns([3, 1])
        nur = k1.multiselect(
            "Nur in diesen Dokumenten suchen (leer = alle)",
            sorted(docs_alle), placeholder="Alle Dokumente",
        )
        k2.caption(f"{len(docs_alle)} Dokument(e) · "
                   f"{sum(docs_alle.values())} Abschnitte")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None

    for i, m in enumerate(st.session_state.history):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("quellen"):
                with st.expander(f"Belegstellen ({len(m['quellen'])})"):
                    for s in m["quellen"]:
                        st.markdown(
                            f"**{s['source']}** · {s['page']} · "
                            f"Relevanz {s['score']}")
                        if s.get("text"):
                            st.caption(s["text"].replace("\n", " ")[:500] + " …")

    # Platzhalter VOR dem Eingabefeld: die neue Frage und die streamende
    # Antwort werden hier hineingeschrieben und stehen damit an der
    # richtigen Stelle, nicht unterhalb des Eingabefelds.
    platz = st.container()

    frage = st.chat_input("z.B. Was haben wir laut Rechnung X am Tag Y für "
                          "Produkt Z gezahlt?")
    if frage:
        if not api_key:
            st.error("Bitte zuerst links den Claude API-Key eintragen.")
            st.stop()

        st.session_state.history.append({"role": "user", "content": frage})
        vorher = st.session_state.history[:-1]

        with platz:
            with st.chat_message("user"):
                st.markdown(frage)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Durchsuche Dokumente ..."):
                        context, quellen, hinweis = belegstellen(
                            frage, api_key, verlauf=vorher,
                            nur_quellen=nur if docs_alle and nur else None,
                        )
                    if hinweis:
                        st.markdown(hinweis)
                        antwort, verbrauch = hinweis, datensatz("frage", MODELL, None)
                    else:
                        # Streaming: die Antwort erscheint Wort für Wort,
                        # statt dass 20 Sekunden nur ein Spinner läuft.
                        ergebnis: dict = {}
                        antwort = st.write_stream(
                            antwort_stream(frage, context, vorher, api_key,
                                           ergebnis)
                        )
                        verbrauch = datensatz("frage", MODELL,
                                              ergebnis.get("usage"))
                        if quellen:
                            with st.expander(f"Belegstellen ({len(quellen)})"):
                                for s in quellen:
                                    st.markdown(
                                        f"**{s['source']}** · {s['page']} · "
                                        f"Relevanz {s['score']}")
                                    if s.get("text"):
                                        st.caption(
                                            s["text"].replace("\n", " ")[:500]
                                            + " …")

                    st.session_state.history.append(
                        {"role": "assistant", "content": antwort,
                         "quellen": quellen})
                    nutzung_speichern(verbrauch)

                    # Unterhaltung sichern. Titel ist die erste Frage – eine
                    # KI-generierte Überschrift kostet Tokens und bringt
                    # hier nichts.
                    import uuid as _uuid

                    if not st.session_state.chat_id:
                        st.session_state.chat_id = str(_uuid.uuid4())
                    chat_speichern({
                        "id": st.session_state.chat_id,
                        "titel": st.session_state.history[0]["content"][:60],
                        "zuletzt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "verlauf": st.session_state.history,
                    })
                except Exception as e:  # noqa: BLE001
                    st.error(f"Fehler: {e}")
                    st.session_state.history.append(
                        {"role": "assistant", "content": f"Fehler: {e}"})


# -------------------------------- Nutzung -----------------------------------

with tab_nutzung:
    st.markdown('<div class="ln-section">Verbrauch und Kosten</div>',
                unsafe_allow_html=True)
    st.caption("Jede Frage und jede Indexierung verbraucht Tokens beim "
               "KI-Anbieter. Die Zahlen stammen aus der Antwort der "
               "Schnittstelle, nichts davon ist geschätzt.")
    render_nutzung(nutzung_laden(), einstellungen_laden, einstellungen_speichern)
