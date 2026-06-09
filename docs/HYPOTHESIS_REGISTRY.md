# HYPOTHESIS REGISTRY

Registro formal de todas las hipótesis de investigación del Attention Observatory.

Cada hipótesis documenta: enunciado, refutación, métricas, tests estadísticos, criterios de aceptación y estado actual.

---

## H1 — Power Law Universal

| Campo | Valor |
|-------|-------|
| **ID** | H1 |
| **Enunciado** | La distribución de atención sigue una ley de potencia en todos los dominios (tecnología, ciencia, deporte, música, política, economía). |
| **H₀** | La distribución de atención no se distingue de una lognormal o exponencial. |
| **H₁** | La distribución de atención es consistente con una ley de potencia (Pareto). |
| **Métricas** | ER, followers, citas, streams, oyentes |
| **Test** | Power Law fit (MLE, Clauset 2009) + Likelihood Ratio Test (Vuong) vs lognormal y exponential |
| **α mínimo** | α > 2.0 para cola pesada (Pareto); α ∈ (2, 3) para regímenes típicos |
| **Criterio** | LR test p < 0.05 Y α > 2.0 en ≥ 4 de 6 dominios |
| **Estado** | PENDIENTE — solo testado en technology (HN α=2.69, Bluesky α=2.29) y science (Wikipedia α no estimado) |
| **Dominios testados** | technology (parcial) |
| **Ultima ejecución** | 2026-06-09 |

---

## H2 — Escalamiento de Desigualdad

| Campo | Valor |
|-------|-------|
| **ID** | H2 |
| **Enunciado** | La desigualdad de atención (Gini) aumenta con el tamaño del ecosistema. |
| **H₀** | No existe correlación entre Gini y N (número de actores). |
| **H₁** | Gini crece monotónicamente con el tamaño del ecosistema. |
| **Métricas** | Gini, bootstrap CI, N actores |
| **Test** | Spearman ρ(Gini, N) con IC 95% bootstrap |
| **Criterio** | ρ > 0.5 con p < 0.05 |
| **Estado** | PENDIENTE — requiere múltiples ejecuciones con distinto N |
| **Nota** | Necesario ejecutar pipeline con distintas configuraciones de muestreo |

---

## H3 — Captura por Super-Hubs

| Campo | Valor |
|-------|-------|
| **ID** | H3 |
| **Enunciado** | Los super-hubs (Z > 3) capturan una fracción desproporcionada de la atención total, independientemente del dominio. |
| **H₀** | La fracción de atención capturada por super-hubs no difiere de lo esperado por azar. |
| **H₁** | Los super-hubs concentran > 30% de la atención total siendo < 1% de los actores. |
| **Métricas** | Top 1% share, Top 5% share, Palma ratio, HHI, Rich Club |
| **Test** | Bootstrap CI para top1_share; comparación contra baseline uniforme |
| **Criterio** | Top 1% share > 0.20 (20%) en ≥ 4 dominios |
| **Estado** | PARCIAL — technology: Top 1% = 30.0%, Top 10% = 60.8%, HHI = 309.3 |
| **Ultima ejecución** | 2026-06-09 |

---

## H4 — Independencia del Contenido

| Campo | Valor |
|-------|-------|
| **ID** | H4 |
| **Enunciado** | La concentración de atención es independiente del contenido (el sentimiento del texto no predice el engagement). |
| **H₀** | Existe correlación significativa entre Sentiment y ER. |
| **H₁** | No existe correlación (o es despreciable) entre Sentiment y ER. |
| **Métricas** | Sentiment avg, ER mean |
| **Test** | Spearman ρ(Sentiment, ER) con p < 0.05 |
| **Criterio** | |ρ| < 0.1 O p > 0.05 (no significativo) |
| **Estado** | CONFIRMADO — ρ = 0.045, p = 0.13 (no significativo) |
| **Ultima ejecución** | 2026-06-09 |

---

## H5 — Mérito Comparable, Desigualdad Similar

| Campo | Valor |
|-------|-------|
| **ID** | H5 |
| **Enunciado** | La estructura de desigualdad emerge incluso cuando los actores poseen méritos comparables (misma antigüedad, mismo tamaño inicial). |
| **H₀** | La desigualdad desaparece al controlar por tamaño de audiencia. |
| **H₁** | La desigualdad persiste intra-cohorte. |
| **Métricas** | Gini intra-cohorte, Partial correlation (controlando followers) |
| **Test** | Partial Spearman ρ(ER, PPI | followers); Gini por decil de followers |
| **Criterio** | Partial ρ significativo Y Gini intra-cohorte > 0.5 |
| **Estado** | PENDIENTE — requiere implementación de cohortes por tamaño/antigüedad |
| **Dependencia** | Partial correlation disponible en `src/stats/validation.py` (pingouin) |

---

## H6 — Propiedad Estructural, no Psicológica

| Campo | Valor |
|-------|-------|
| **ID** | H6 |
| **Enunciado** | La hipercentralización es una propiedad estructural emergente de la red, no atribuible a psicología individual. |
| **H₀** | La centralización de red no se correlaciona con la desigualdad de engagement. |
| **H₁** | La centralización de red (Freeman) y la desigualdad (Gini) están correlacionadas positivamente. |
| **Métricas** | Freeman centralization, Gini, Modularity, Assortativity |
| **Test** | Spearman ρ(Centralization, Gini); comparación contra redes aleatorias (Erdős–Rényi) |
| **Criterio** | ρ > 0.5 Y modularidad > 0.3 |
| **Estado** | PENDIENTE — requiere múltiples ejecuciones de red con distinto muestreo |
| **Dependencia** | Network metrics disponibles en `src/stats/network.py` |

---

## HC — Hipótesis Central (Unificada)

| Campo | Valor |
|-------|-------|
| **ID** | HC |
| **Enunciado** | La hipercentralización de la atención es una ley emergente universal de los sistemas abiertos de información humana, independiente del dominio. |
| **H₀** | La concentración extrema es un artefacto de dominios específicos (tecnología, redes sociales). |
| **H₁** | El patrón Gini > 0.85, α ≈ 2–3, super-hubs presentes se replica en ≥ 6 dominios no relacionados. |
| **Criterio** | H1–H6 confirmadas en ≥ 6 dominios (technology, science, sports, music, entertainment, politics, economy, health) |
| **Estado** | PENDIENTE — 1 dominio parcial (technology), 1 dominio sin testar (science) |
| **Próximo paso** | Incorporar OpenAlex (ciencia) y Reddit (tecnología ampliada) |

---

## Resumen de Estado

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1 Power Law Universal | 🟡 Parcial | Technology α=2.11; falta science, sports, music |
| H2 Escalamiento | 🔴 Pendiente | Requiere múltiples ejecuciones |
| H3 Captura Super-Hubs | 🟡 Parcial | Top 1%=30.0% en technology |
| H4 Independencia Contenido | 🟢 Confirmado | ρ=0.045, p=0.13 |
| H5 Mérito Comparable | 🔴 Pendiente | Requiere cohortes |
| H6 Propiedad Estructural | 🔴 Pendiente | Requiere redes multi-ejecución |
| HC Universal | 🔴 Pendiente | Depende de H1–H6 en ≥ 6 dominios |
