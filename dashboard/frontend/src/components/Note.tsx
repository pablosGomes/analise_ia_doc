import type { PropsWithChildren } from "react";

interface NoteProps {
  variante: "ok" | "warn" | "bad";
  titulo: string;
}

/** Bloco de aviso com barra de acento lateral — usado para os alertas de
 * atalho (veredito) e de diversidade de geração (testes de generalização). */
export function Note({ variante, titulo, children }: PropsWithChildren<NoteProps>) {
  return (
    <div className={`note ${variante}`}>
      <span className="mk" />
      <div>
        <b className="note-titulo">{titulo}</b>
        <span>{children}</span>
      </div>
    </div>
  );
}
