// @ts-types="@/types/jinja.d.ts"
import {
    ArrayLiteral,
    Identifier,
    Literal,
    SetStatement,
    Template,
} from "@huggingface/jinja";
import * as z from "@/common/myZod.ts";
import * as Path from "@std/path";

// From @huggingface/jinja
export function range(start: number, stop?: number, step = 1): number[] {
    if (stop === undefined) {
        stop = start;
        start = 0;
    }

    const result: number[] = [];
    for (let i = start; i < stop; i += step) {
        result.push(i);
    }
    return result;
}

const TemplateMetadataSchema = z.object({
    stop_strings: z.array(z.string()).default([]),
    tool_start: z.string().optional(),
});

type TemplateMetadata = z.infer<typeof TemplateMetadataSchema>;

export class PromptTemplate {
    name: string;
    rawTemplate: string;
    template: Template;
    metadata: TemplateMetadata;

    public constructor(
        name: string,
        rawTemplate: string,
    ) {
        this.name = name;
        this.rawTemplate = rawTemplate;
        this.template = new Template(rawTemplate);
        this.metadata = this.extractMetadata(this.template);
    }

    private assignMetadataValue<K extends keyof TemplateMetadata>(
        metadata: TemplateMetadata,
        key: K,
        value: unknown,
    ) {
        metadata[key] = value as TemplateMetadata[K];
    }

    private extractMetadata(template: Template) {
        const metadata: TemplateMetadata = TemplateMetadataSchema.parse({});

        template.parsed.body.forEach((statement) => {
            if (statement.type === "Set") {
                const setStatement = statement as SetStatement;

                const assignee = setStatement.assignee as Identifier;
                const foundMetaKey = Object.keys(TemplateMetadataSchema.shape)
                    .find(
                        (key) => key === assignee.value,
                    ) as keyof TemplateMetadata;

                if (foundMetaKey) {
                    const fieldSchema =
                        TemplateMetadataSchema.shape[foundMetaKey];

                    let result: unknown;
                    if (setStatement.value.type === "ArrayLiteral") {
                        const arrayValue = setStatement.value as ArrayLiteral;
                        result = arrayValue.value.map((e) => {
                            const literalValue = e as Literal<unknown>;
                            return literalValue.value;
                        });
                    } else if (setStatement.value.type.endsWith("Literal")) {
                        const literalValue = setStatement.value as Literal<
                            unknown
                        >;
                        result = literalValue.value;
                    }

                    const parsedValue = fieldSchema.safeParse(result);
                    if (parsedValue.success) {
                        this.assignMetadataValue(
                            metadata,
                            foundMetaKey,
                            parsedValue.data,
                        );
                    }
                }
            }
        });

        return metadata;
    }

    static async fromFile(templatePath: string) {
        const parsedPath = Path.parse(templatePath);
        parsedPath.ext = ".jinja";
        const formattedPath = Path.format({ ...parsedPath, base: undefined });
        const rawTemplate = await Deno.readTextFile(formattedPath);
        return new PromptTemplate(parsedPath.name, rawTemplate);
    }
}
