"""
consent.py — Aviso de tratamiento de información y Términos y Condiciones,
compartidos entre la UI de Streamlit (src/ui/app.py) y el bot de Telegram
(src/bot/maternas_bot.py).

Dos capas de aceptación SECUENCIALES, ambas obligatorias antes de permitir
cualquier conversación, en cada sesión nueva:
  1. CONSENT_TEXT — tratamiento de datos (qué se guarda, por qué, cómo
     retirarse).
  2. TERMS_TEXT   — Términos y Condiciones de uso (naturaleza del proyecto,
     límites de responsabilidad, propiedad del contenido, uso aceptable).
Si el usuario rechaza cualquiera de las dos, la sesión termina — la próxima
vez que escriba se le vuelve a mostrar la capa 1 desde cero (ver
consent_gate.py y maternas_bot.py).

Texto en plano (sin markdown) para que se renderice igual en Streamlit y en
Telegram, evitando errores de parseo de entidades en Telegram.
"""

CONSENT_TEXT = (
    "⚠️ AVISO SOBRE TRATAMIENTO DE INFORMACIÓN\n\n"
    "Este chatbot es un proyecto de INVESTIGACIÓN en FASE EXPERIMENTAL. Usa "
    "inteligencia artificial y NO SUSTITUYE la orientación de un profesional "
    "competente.\n\n"
    "Podemos conservar tu NOMBRE PREFERIDO O ALIAS y un CÓDIGO INTERNO de "
    "identificación. NO escribas tu nombre legal completo, número de "
    "documento, dirección, contraseñas ni datos financieros.\n\n"
    "Tu conversación se procesa para mantener el CONTEXTO del diálogo y los "
    "fines del proyecto de investigación. Si se usa en análisis o "
    "publicaciones, se aplican medidas de ANONIMIZACIÓN o SEUDONIMIZACIÓN.\n\n"
    "Tu participación es VOLUNTARIA: puedes NO responder preguntas sensibles "
    "y puedes SOLICITAR EL RETIRO de tu información en cualquier momento.\n\n"
    "Este es un PROTOTIPO DE INVESTIGACIÓN — aún no apto para uso clínico, "
    "comercial o productivo real.\n\n"
    "¿Aceptas continuar bajo estas condiciones?"
)

FAREWELL_TEXT = (
    "Entendido — no se procesará tu conversación. Sesión finalizada. 👋\n"
    "Si cambias de opinión, escríbeme de nuevo y volveré a mostrarte este aviso."
)

# ---------------------------------------------------------------------------
# Términos y Condiciones — BORRADOR, no es texto legal definitivo
# ---------------------------------------------------------------------------
# Redactado como especificación funcional razonable, consistente con el
# resto del proyecto (mismo tono que CONSENT_TEXT y la sección "Advertencia
# de uso y puesta en producción" del README) — NO reemplaza la revisión y
# aprobación de las instancias jurídicas y éticas correspondientes, que el
# propio README ya exige como requisito antes de cualquier uso productivo.
# No completar/activar en un despliegue real sin ese visto bueno.
# ---------------------------------------------------------------------------

TERMS_TEXT = (
    "📄 TÉRMINOS Y CONDICIONES DE USO (BORRADOR — PENDIENTE DE REVISIÓN LEGAL)\n\n"
    "Este texto es una especificación funcional preliminar, no un documento "
    "legal definitivo. Su contenido debe ser revisado y aprobado por las "
    "instancias jurídicas y éticas del proyecto antes de cualquier uso "
    "productivo (ver README, sección \"Advertencia de uso y puesta en "
    "producción\").\n\n"
    "1. NATURALEZA DEL SERVICIO: Maternas es un prototipo de investigación "
    "desarrollado en el marco de la Convocatoria 890 de Minciencias y la "
    "Institución Universitaria de Envigado. No es un dispositivo médico, no "
    "presta servicios de salud y no sustituye el diagnóstico, tratamiento "
    "ni consejo de un profesional competente.\n\n"
    "2. SIN GARANTÍAS: El servicio se ofrece \"tal cual\", en fase "
    "experimental, sin garantía de disponibilidad, exactitud, vigencia ni "
    "idoneidad para un fin específico. Las respuestas se generan con "
    "inteligencia artificial y pueden contener errores.\n\n"
    "3. LÍMITE DE RESPONSABILIDAD: El equipo del proyecto no asume "
    "responsabilidad por decisiones tomadas a partir de las respuestas del "
    "chatbot. Ante cualquier urgencia o duda médica real, contacta a un "
    "profesional de salud o a los servicios de emergencia.\n\n"
    "4. USO ACEPTABLE: No debes usar este servicio para suplantar a "
    "terceros, ingresar datos de otra persona sin su consentimiento, ni "
    "para fines distintos a la consulta informativa de salud materna que "
    "motiva este proyecto.\n\n"
    "5. PROPIEDAD Y CONTENIDO: Las respuestas se generan a partir de "
    "fuentes bibliográficas citadas dentro de cada respuesta; su uso está "
    "sujeto a la licencia de cada fuente (ver README). El código del "
    "proyecto y su documentación pertenecen al equipo de investigación.\n\n"
    "6. CAMBIOS: Estos términos pueden actualizarse mientras el proyecto "
    "esté en fase experimental; la versión vigente es la que se muestra al "
    "inicio de cada sesión nueva.\n\n"
    "¿Aceptas continuar bajo estos Términos y Condiciones?"
)

TERMS_ACCEPTED_TEXT = "✅ Gracias — aceptaste los Términos y Condiciones. Ya puedes escribirme tu pregunta."
