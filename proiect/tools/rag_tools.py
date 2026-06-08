"""Tool-uri pe documente — conectează agentul (L1-L2) la documentele din pgvector (L4).

Două tool-uri complementare:
  - `search_documents` — căutare SEMANTICĂ în textul liber (clauze, "ce scrie despre…").
  - `query_documents`  — date STRUCTURATE extrase (numere, sume, total, părți, date).

Respectă convenția cursului: tool-uri înregistrate prin `@register_tool` (Registry Pattern),
NU prin `@tool`-ul LangChain. Ajung la agent prin `ToolWrapper.catalog()` → `bind_tools`.
Importurile DB/RAG sunt lazy în corpul funcției, ca pachetul `tools` să rămână ușor de
importat chiar dacă stiva de storage nu e instalată/pornită.
"""

from .params_models import QueryDocumentsParams, SearchDocumentsParams
from .registry import register_tool


@register_tool
def search_documents(params: SearchDocumentsParams) -> str:
    """Caută semantic în documentele încărcate (facturi, contracte) și returnează
    fragmentele relevante cu sursă și scor. Folosește pentru întrebări despre
    conținutul documentelor: clauze, sume, termene de plată, părți contractante."""
    from database import transaction
    from rag_service import RAGService

    try:
        with transaction() as db:
            rag = RAGService(db)
            context = rag.get_context(params.query, top_k=params.top_k)
    except Exception as e:
        return f"Eroare căutare documente: {e}"

    return context or "Nu am găsit documente relevante pentru această întrebare."


@register_tool
def query_documents(params: QueryDocumentsParams) -> str:
    """Returnează datele STRUCTURATE extrase din documente (facturi/contracte): numere,
    sume (subtotal, TVA, total), monedă, părți (furnizor/client, prestator/beneficiar),
    date și durată. Folosește pentru valori exacte și agregări (totaluri, comparații);
    pentru clauze sau text liber folosește `search_documents`."""
    import json

    from database import transaction
    from repositories import DocumentRepository

    doc_type = params.doc_type.strip().lower() or None
    needle = params.contains.strip().lower()

    try:
        with transaction() as db:
            docs = DocumentRepository(db).get_documents(doc_type=doc_type)
            linii = []
            for d in docs:
                numar = (d.extracted or {}).get("numar", "")
                if needle and needle not in d.filename.lower() and needle not in str(numar).lower():
                    continue
                payload = (
                    json.dumps(d.extracted, ensure_ascii=False)
                    if d.extracted
                    else "(extragere structurată indisponibilă)"
                )
                linii.append(f"[{d.filename} · {d.doc_type} {numar}] {payload}")
    except Exception as e:
        return f"Eroare interogare documente: {e}"

    if not linii:
        return "Nu am găsit documente cu date structurate pentru acest filtru."
    return "\n\n".join(linii)
