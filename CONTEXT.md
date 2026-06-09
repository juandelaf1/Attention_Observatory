# ATTENTION OBSERVATORY — MARCO CIENTÍFICO

## PREGUNTA CENTRAL

**¿La hipercentralización de la atención es una ley universal de los sistemas humanos de atención, o una propiedad específica de ciertos ecosistemas digitales?**

---

## HIPÓTESIS FUNDAMENTALES (H1–H6)

| ID | Hipótesis | Contraste empírico | Fuente espec |
|----|-----------|-------------------|--------------|
| **H1** | La distribución de atención sigue una ley de potencia en todos los dominios. | Power Law fit + LR test (vs lognormal, exponential) por dominio | Radical §II.1 |
| **H2** | La desigualdad de atención aumenta con el tamaño del ecosistema. | Correlación Gini vs N actores; bootstrap CI | Inviabilidad §I |
| **H3** | Los super-hubs capturan una fracción desproporcionada de la atención total independientemente del dominio. | Top 1%/5%/10% share; Palma ratio; HHI | Radical §II |
| **H4** | La concentración es independiente del contenido (no correlación Sentiment–ER). | Spearman Sentiment vs ER; Kruskal-Wallis entre dominios | Inviabilidad §III-C |
| **H5** | La estructura emerge incluso cuando los actores poseen méritos comparables. | Gini intra-cohortes de misma antigüedad/tamaño | Radical §II.2 |
| **H6** | La hipercentralización es una propiedad estructural emergente de la red, no psicológica. | Network centralization (Freeman) vs Gini correlación | Inviabilidad §IV |

---

## HIPÓTESIS CENTRAL (UNIFICADA)

**H0**: La concentración extrema de atención es una consecuencia inevitable de la evolución de sistemas abiertos de información, independientemente del dominio (tecnología, ciencia, deporte, música, política, economía).

**Evidencia requerida**: Consistencia del patrón Gini > 0.85, Power Law α ≈ 2–3, super-hubs presentes, en ≥ 6 dominios no relacionados.

---

## DOMINIOS Y TAXONOMÍA DE PLATAFORMAS

| Domain | platform_type | Fuentes actuales | Fuentes futuras |
|--------|--------------|------------------|-----------------|
| **technology** | social_network | HackerNews, Bluesky, Mastodon | Reddit tech |
| **technology** | collaborative_network | GitHub | GitLab |
| **technology** | corpus_dataset | HuggingFace | — |
| **science** | knowledge_network | Wikipedia | OpenAlex, arXiv, Semantic Scholar, Crossref |
| **sports** | social_network | — | Instagram, X/Twitter, Wikidata |
| **music** | streaming | — | Spotify Charts, LastFM, MusicBrainz |
| **entertainment** | database | — | TMDB, IMDb |
| **politics** | knowledge_network | — | Wikidata, ParlGov, GDELT |
| **economy** | database | — | Crunchbase, OpenCorporates |

**Regla**: HuggingFace (corpus_dataset) no participa en métricas de desigualdad — solo NLP.

---

## ARQUITECTURA

```
BRONZE (raw) ──> SILVER (schema canónico) ──> GOLD (métricas agregadas)
     │                    │                            │
  actores + posts     columnas canónicas          vectores [ER, PPI,
  por fuente          + domain + platform_type     Sentiment, AFI]
                                                    + stats + snapshot
```

### Silver schema canónico

```python
actor_id: str
platform: str
domain: str                     # technology, science, sports, ...
platform_type: str              # social_network, knowledge_network, ...
attention: float                # followers / subscribers / citations
engagement: float               # ER o equivalente
content_count: int              # posts / revisions / commits
timestamp: datetime
```

---

## MÉTRICAS DE CONCENTRACIÓN

| Métrica | Definición | Propósito |
|---------|-----------|-----------|
| **Gini** | 2Σ(i·v_i) / (n·Σv) - (n+1)/n | Desigualdad general |
| **Gini bootstrap** | IC 95% por remuestreo | Significancia de diferencias |
| **HHI** | Σ(p_i²) · 10000 | Concentración tipo mercado |
| **Shannon Entropy** | -Σ(p_i · ln p_i) | Diversidad efectiva |
| **Effective N** | 1 / Σ(p_i²) | Número equivalente de actores |
| **Palma Ratio** | Top 10% / Bottom 40% | Polarización |
| **Top Share** | Top 1%, 5%, 10% | Captura por élite |
| **Rich Club** | φ(k) / φ_random | Conexión entre super-hubs |
| **Freeman Centralization** | Σ(C_max - C_i) / max possible | Centralización de red |

---

## VALIDACIÓN ESTADÍSTICA

| Test | Aplicación |
|------|-----------|
| **Power Law vs Lognormal** | Likelihood Ratio Test (Vuong) | Validar H1 |
| **Bootstrap Gini** | IC 95% por dominio | Validar H2 |
| **Kruskal-Wallis** | Diferencia entre dominios/plataformas | Validar H4 |
| **Dunn Post-Hoc** | Pares significativos | Desglose KW |
| **Spearman / Pearson** | Correlaciones entre features | Validar H4 |
| **Partial Correlation** | Controlando tamaño de audiencia | Validar H5 |

---

## MÉTRICAS LONGITUDINALES (dinámica)

| Métrica | Definición |
|---------|-----------|
| dGini/dt | Velocidad de concentración |
| dAlpha/dt | Deriva de cola pesada |
| Mobility Index | Probabilidad de entrar al top 1%/5%/10% |
| Hub Persistence | Tiempo medio de dominio de super-hubs |
| Entrant Survival | Fracción de nuevos actores que persisten |
| Attention Half-Life | Tiempo para perder 50% de atención acumulada |

---

## MÉTRICAS DE RED

| Métrica | Definición |
|---------|-----------|
| Degree | Conexiones directas |
| Betweenness | Puentes estructurales |
| Eigenvector | Influencia por vecinos |
| PageRank | Importancia por enlaces entrantes |
| Closeness | Distancia media a otros nodos |
| Assortativity | Homofilia (conectados similares) |
| Modularity | Estructura de comunidades |
| K-Core | Robustez del núcleo |
| Rich Club | Concentración entre hubs |

---

## REPRODUCIBILIDAD

- Mismo pipeline → mismos resultados (determinístico)
- Snapshots JSON por ejecución (`data/executions/`)
- Dataset hash planeado para Fase 2
- DVC considerado para escalado futuro

---

## MANIFIESTOS RECTORES (SPEC)

El proyecto se rige por dos documentos fundacionales en `manifiestos/`:

1. **MANIFIESTO_Inviabilidad_Evolutiva_HiperCentralizacion.pdf**
2. **MANIFIESTO_Radical_Pesimismo_Luz_Claridad.pdf**

Ambos constituyen la **especificación** (Spec): de ellos derivamos hipótesis, métricas y capas analíticas.

---

## REGLAS DE NEGOCIO (INMUTABLES)

1. **Neutralidad epistemológica** — describe, no prescribe, no moraliza.
2. **No inferencia psicológica individual** — no etiquetar personas.
3. **Separación datos / interpretación** — datos neutros, interpretación marcada.
4. **Reproducibilidad total** — mismo pipeline → mismos resultados.
5. **Trazabilidad ELT** — cada transformación rastreable Bronze → Silver → Gold.
6. **HuggingFace excluido** de métricas de desigualdad (solo NLP corpus).

---

## FUENTES DE DATOS

| Fuente | Domain | Auth | Estado |
|--------|--------|------|--------|
| Hacker News | technology | Ninguna | ✅ |
| Wikipedia | science | User-Agent | ✅ |
| HuggingFace | technology | Ninguna | ✅ (solo NLP) |
| Bluesky | technology | Ninguna | ✅ |
| Mastodon | technology | Ninguna | ✅ |
| GitHub | technology | Token gratuito | ✅ |
| YouTube | technology | API key | ✅ |
| Telegram | technology | Bot token | ⏳ |
| Reddit | technology | OAuth | ❌ |
| OpenAlex | science | Ninguna | 🔲 |
| arXiv | science | Ninguna | 🔲 |
| Spotify | music | API key | 🔲 |
| TMDB | entertainment | API key | 🔲 |

---

## ESTRUCTURA DEL REPOSITORIO

```
attention_observatory/
├── manifiestos/             # Spec: documentos fundacionales
├── img/                     # Visualizaciones generadas
├── src/
│   ├── ingesta/             # Conectores por fuente
│   ├── transform/           # silver_to_gold.py
│   ├── stats/               # inequality.py, network.py, validation.py
│   ├── nlp/                 # sentiment.py
│   └── analysis/            # eda.py, research.py, longitudinal.py
├── data/
│   ├── bronze/              # Raw parquet
│   ├── silver/              # Schema canónico
│   ├── gold/                # fact_metrics.parquet
│   ├── reports/             # EDA + research JSONs
│   └── executions/          # Snapshots longitudinales
├── main.py                  # Orquestador
├── app.py                   # Dashboard
├── generate_charts.py       # Visualizaciones
└── docs/                    # Reportes
```

---

## PRIORIDAD DE EJECUCIÓN

| Fase | Qué | Por qué |
|------|-----|---------|
| 1 | Reformular hipótesis + taxonomía | Base conceptual correcta |
| 2 | Validación estadística completa | Rigor científico |
| 3 | Nuevas métricas de concentración | Medir lo que importa |
| 4 | Network analysis central | Probar H6 |
| 5 | Silver layer real | Reproducibilidad |
| 6 | OpenAlex + arXiv | Dominio ciencia |
| 7 | Reddit | Dominio tecnología completo |
| 8 | Deporte, música, entretenimiento | Test universal |

---

*"El objetivo no es reformar la máquina o lanzar un lamento ético sobre el estado de la cultura. El objetivo es documentar su autopsia con la precisión de la física de partículas."*
