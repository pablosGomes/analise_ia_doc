# scripts/anonimizacao/

Responsável por mascarar/pseudonimizar dados sensíveis (CPF, RG, nome, assinatura,
foto de rosto, impressão digital) nos documentos legítimos antes de qualquer
commit ou uso em treino, conforme a base legal de consentimento descrita em
`consentimento/TERMO_CONSENTIMENTO_MODELO.md` e na Seção 6 da documentação técnica.

Ponto de partida: reaproveitar e adaptar `anonimizar_documento.py` do projeto
`prototipo_Ia_doc` (união de detecção de rosto via MediaPipe + Haar Cascade em
4 rotações, mascaramento com cor local + ruído gaussiano, sidecar `.mascaras.json`
para permitir excluir a região da máscara de features de treino).

Regra inegociável: nenhum documento entra em `datasets/legitimos/` sem antes
passar por este pipeline e ser verificado manualmente.
