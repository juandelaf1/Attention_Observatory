# Atención Observatory — Reporte Ciudadano

## ¿Qué es esto?

Un proyecto que **mide cómo se distribuye la atención en internet**. No juzga si está bien o mal. Solo observa y mide, como un termómetro mide la fiebre.

La atención humana es limitada: tenemos 24 horas al día, y nuestro cerebro solo puede procesar cierta cantidad de información. Las plataformas digitales (YouTube, redes sociales, foros) compiten por ese recurso escaso. Nosotros medimos cómo se reparte.

## ¿Qué encontramos?

### 1. La atención es extremadamente desigual

Imagina una fiesta donde **el 1% de los invitados se queda con el 98% de la comida**. Eso es lo que pasa en internet con la atención. Un puñado de creadores, canales y cuentas concentran casi toda la atención del público.

El indicador que usamos (Gini) da **0.97 sobre 1**. Para contexto:
- 0.00 sería reparto perfecto (todos reciben lo mismo)
- 0.50 es desigualdad alta (como en países con mucha desigualdad económica)
- **0.97 es desigualdad extrema**

### 2. Es un fenómeno natural de las redes

No es "culpa" de nadie. Las redes funcionan así: quien ya tiene atención, recibe más atención. Es como la fama: mientras más famoso eres, más fácil es seguir siendo famoso. Esto se llama **"ley de potencia"** y ocurre en todos lados — en la música, el cine, los libros, y ahora en internet.

### 3. Cada plataforma es distinta

| Plataforma | ¿Qué tan desigual es? |
|-----------|----------------------|
| Mastodon | Muy desigual (0.97) |
| GitHub | Muy desigual (0.96) |
| Hacker News | Muy desigual (0.90) |
| Bluesky | Desigual (0.87) |

Todas son desiguales, pero algunas más que otras. Las redes más "alternativas" (Mastodon, Bluesky) también terminan siendo desiguales — es parte de cómo funcionan las redes.

### 4. El contenido emocional no importa tanto

Medimos si los mensajes positivos o negativos tienen más atención. **No hay diferencia**. Lo que importa no es lo que dices, sino quién eres y cuánta audiencia ya tienes.

### 5. El "prestigio" y la presión de publicar

Encontramos que **quien usa palabras más sofisticadas (referencias académicas, lenguaje de prestigio) suele publicar con menos frecuencia**. Es como si hubiera dos estrategias:
- Publicar mucho y llegar a mucha gente (volumen)
- Publicar poco pero con calidad y prestigio (exclusividad)

Hay una relación negativa débil pero real entre cuánto publicas y qué tan "prestigioso" es tu contenido.

## ¿Qué plataformas analizamos?

Analizamos **7 fuentes** de datos reales:

- **Hacker News** — foro de tecnología
- **Wikipedia** — edits de artículos
- **Bluesky** — red social descentralizada
- **Mastodon** — red social federada
- **GitHub** — plataforma de código
- **YouTube** — videos y canales
- **HuggingFace** — dataset de emociones en texto

Cada vez que ejecutamos el análisis, guardamos una foto de cómo están las cosas. Así podemos ver si la desigualdad aumenta o disminuye con el tiempo.

## ¿Para qué sirve esto?

No es para decir si algo está bien o mal. Es para **entender cómo funciona el sistema**. Así como un ecólogo estudia un ecosistema sin juzgar a los animales que viven allí, nosotros estudiamos el ecosistema digital.

El objetivo es poder detectar **cuándo el sistema está llegando a su límite**: cuando la gente está tan cansada que empieza a abandonar las plataformas, o cuando la concentración es tan extrema que el sistema se vuelve frágil.

## Una nota importante

Este sistema **no analiza personas**. No dice "tal persona es mala" o "tal creador acapara la atención". Analiza **estructuras**, no individuos. Como medir la temperatura del océano, no juzgar al pez que nada en él.

---

_Este reporte fue generado por Attention Observatory — un sistema de modelado empírico de la economía de la atención digital. Los datos provienen de APIs públicas. No se almacena información personal ni se rastrean individuos._
