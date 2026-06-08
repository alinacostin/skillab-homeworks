"""
Scheme Pydantic pentru extracția structurată 
"""

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """O linie de produs/serviciu dintr-o factură."""

    denumire: str = Field(description="Denumirea produsului sau serviciului")
    cantitate: float = Field(default=1, description="Cantitatea")
    valoare: float = Field(default=0, description="Valoarea totală a liniei, în moneda facturii")


class Invoice(BaseModel):
    """Factură fiscală."""

    numar: str = Field(description="Numărul facturii (ex: FV-2024-001)")
    data: str = Field(description="Data emiterii, normalizată la YYYY-MM-DD")
    furnizor: str = Field(description="Numele furnizorului")
    furnizor_cui: str = Field(default="", description="CUI-ul furnizorului (ex: RO12345678)")
    client: str = Field(description="Numele clientului")
    client_cui: str = Field(default="", description="CUI-ul clientului")
    subtotal: float = Field(default=0, description="Subtotalul fără TVA")
    tva: float = Field(default=0, description="Valoarea TVA")
    total: float = Field(description="Totalul de plată, cu TVA")
    moneda: str = Field(default="RON", description="Moneda (RON, EUR)")
    termen_plata: str = Field(default="", description="Termenul de plată (ex: '30 zile de la emitere')")
    produse: list[LineItem] = Field(
        default_factory=list, description="Liniile de produse/servicii facturate"
    )


class Contract(BaseModel):
    """Contract de prestări servicii / consultanță."""

    numar: str = Field(description="Numărul contractului (ex: CC-2024-008)")
    data_incheiere: str = Field(description="Data încheierii, normalizată la YYYY-MM-DD")
    prestator: str = Field(description="Numele prestatorului / consultantului")
    beneficiar: str = Field(description="Numele beneficiarului / clientului")
    valoare: float = Field(description="Valoarea totală a contractului, fără TVA")
    moneda: str = Field(default="RON", description="Moneda (RON, EUR)")
    durata_luni: int = Field(default=0, description="Durata contractului în luni")
    obiect: str = Field(default="", description="Obiectul contractului, pe scurt")
    obligatii_prestator: list[str] = Field(
        default_factory=list, description="Lista principalelor obligații ale prestatorului"
    )


# Registry tip document → schemă (folosit de pipeline.py)
SCHEMA_BY_TYPE: dict[str, type[BaseModel]] = {
    "factura": Invoice,
    "contract": Contract,
}
