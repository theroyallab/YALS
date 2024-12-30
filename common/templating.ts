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

const TemplateMetadataSchema = z.object({
    stop_strings: z.array(z.string()).optional(),
    tool_start: z.string().optional(),
    tool_start_token: z.number().optional(),
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

    public render(context: Record<string, unknown> = {}): string {
        return this.template.render(context);
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
                        // deno-lint-ignore no-explicit-any
                        metadata[foundMetaKey] = parsedValue.data as any;
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
