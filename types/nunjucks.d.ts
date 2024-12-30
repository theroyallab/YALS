// Extended types for nunjucks
// These are the bare minimum to parse out template metadata through the AST

// deno-lint-ignore-file no-explicit-any no-unused-vars
// @ts-types="@types/nunjucks"
import nunjucks from "nunjucks";
export * from "nunjucks"

declare module "nunjucks" {
    export interface Node {
        lineno: number;
        colno: number;
        value?: any;
        children?: Node[];
    }

    export interface TargetValue extends Node {
        value: string;
    }

    export interface SetNode extends Node {
        targets: TargetValue[];
        value: Node;
    }

    export interface NodeList extends Node {
        children: Node[];
    }

    export interface ParserModule {
        parse(src: string, extensions?: any, opts?: any): Node;
    }

    export const parser: ParserModule;
}
