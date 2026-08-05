import type { PropsWithChildren, ReactNode } from "react";

interface CardProps {
  titulo: string;
  lead?: ReactNode;
  className?: string;
}

/** Cartão padrão de seção: título em caixa alta com régua, texto de apoio
 * opcional, conteúdo livre abaixo. */
export function Card({ titulo, lead, className, children }: PropsWithChildren<CardProps>) {
  return (
    <section className={`card${className ? ` ${className}` : ""}`}>
      <div className="eyebrow">{titulo}</div>
      {lead && <p className="lead">{lead}</p>}
      {children}
    </section>
  );
}
