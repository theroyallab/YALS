// @ts-types="@/types/nunjucks.d.ts"
import nunjucks from "nunjucks";
import * as z from "@/common/myZod.ts";
import * as Path from "@std/path";

const TemplateMetadataSchema = z.object({
    stop_strings: z.array(z.string()).optional(),
    tool_start: z.string().optional(),
    tool_start_token: z.number().optional(),
});

type TemplateMetadata = z.infer<typeof TemplateMetadataSchema>;

function raiseException(message: string) {
    throw new Error(message);
}

export class PromptTemplate {
    name: string;
    rawTemplate: string;
    environment: nunjucks.Environment;
    template: nunjucks.Template;
    metadata: TemplateMetadata;

    public constructor(
        name: string,
        rawTemplate: string,
    ) {
        this.name = name;
        this.rawTemplate = rawTemplate;
        this.environment = nunjucks.configure({ autoescape: false })
            .addGlobal("raise_exception", raiseException);
        this.template = new nunjucks.Template(rawTemplate, this.environment);
        this.metadata = this.extractMetadata(rawTemplate);
    }

    public render(context: object = {}): string {
        return this.template.render(context);
    }

    private extractMetadata(rawTemplate: string): TemplateMetadata {
        const ast = nunjucks.parser.parse(rawTemplate);
        const metadata: TemplateMetadata = TemplateMetadataSchema.parse({});
        if (!ast.children) {
            return metadata;
        }

        ast.children.forEach((node) => {
            // Targets is unique to a setNode
            if ("targets" in node) {
                const setNode = node as nunjucks.SetNode;
                if (setNode.targets.length === 0) {
                    return;
                }
                const foundMetaKey = Object.keys(TemplateMetadataSchema.shape)
                    .find(
                        (key) => key === setNode.targets[0].value,
                    ) as keyof TemplateMetadata;

                if (foundMetaKey) {
                    // Get field schema from overall schema
                    const fieldSchema =
                        TemplateMetadataSchema.shape[foundMetaKey];

                    // Only use for validation. For some reason, the parsed data can't be assigned
                    let result;
                    if (setNode.value.children) {
                        result = setNode.value.children.map((child) =>
                            child.value
                        );
                    } else {
                        result = setNode.value.value;
                    }

                    const parsedValue = fieldSchema.safeParse(result);
                    if (parsedValue.success) {
                        metadata[foundMetaKey] = result;
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
