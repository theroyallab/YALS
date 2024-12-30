export * from "@huggingface/jinja";

declare module "@huggingface/jinja" {
    export class Statement {
        type: string;
    }

    export class Expression extends Statement {
        type: string;
    }

    abstract class Literal<T> extends Expression {
        value: T;
        type: string;
        constructor(value: T);
    }

    export class Identifier extends Expression {
        value: string;
        type: string;

        /**
         * @param {string} value The name of the identifier
         */
        constructor(value: string);
    }

    export class NumericLiteral extends Literal<number> {
        type: string;
    }

    export class StringLiteral extends Literal<string> {
        type: string;
    }

    export class BooleanLiteral extends Literal<boolean> {
        type: string;
    }

    export class NullLiteral extends Literal<null> {
        type: string;
    }

    export class ArrayLiteral extends Literal<Expression[]> {
        type: string;
    }

    export class TupleLiteral extends Literal<Expression[]> {
        type: string;
    }

    export class ObjectLiteral extends Literal<Map<Expression, Expression>> {
        type: string;
    }

    export class SetStatement extends Statement {
        assignee: Expression;
        value: Expression;
        type: string;
        constructor(assignee: Expression, value: Expression);
    }
}
