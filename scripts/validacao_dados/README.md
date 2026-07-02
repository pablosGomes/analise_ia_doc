# scripts/validacao_dados/

Camada de validação barata e determinística, independente da imagem
(Seção 3.4 da documentação técnica):

- `validar_cpf.py` — checagem do dígito verificador via OCR do campo.
- `validar_mrz.py` — leitura e checksum da MRZ (passaportes / alguns RGs).
- `validar_qr.py` — leitura e validação da assinatura digital do QR code
  (novo modelo de CIN/CNH digital).

Qualquer falha aqui sinaliza fraude independentemente do score do modelo de
imagem (regra combinada na Fase C do roadmap).
