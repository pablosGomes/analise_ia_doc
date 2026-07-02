# scripts/anonimizacao/

Responsável por mascarar/pseudonimizar dados sensíveis (CPF, RG, nome, assinatura,
foto de rosto, impressão digital) nos documentos legítimos.

Ponto de partida: reaproveitar e adaptar `anonimizar_documento.py` do projeto
`prototipo_Ia_doc` (união de detecção de rosto via MediaPipe + Haar Cascade em
4 rotações, mascaramento com cor local + ruído gaussiano, sidecar `.mascaras.json`).

NOTA (2026-07-02): por decisão do usuário, o lote atual de `datasets/legitimos/`
está sem anonimização (risco assumido). Esta etapa fica pendente para lotes
futuros ou para quando o usuário decidir aplicar retroativamente.
