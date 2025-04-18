import { hasLlguidance } from "@/bindings/lib.ts";
import { SamplerBuilder } from "@/bindings/samplers.ts";
import { logger } from "@/common/logging.ts";

export class YALSGrammar {
    private sampler: SamplerBuilder;

    constructor(sampler: SamplerBuilder) {
        this.sampler = sampler;
    }

    BNF(grammar: string) {
        if (hasLlguidance) {
            this.sampler.llguidance(grammar);
        } else {
            logger.warn(
                "YALS was not built with LLGuidance. Using GBNF.",
            );

            this.sampler.grammar(grammar);
        }
    }

    jsonSchema(schema: Record<string, unknown>) {
        if (!hasLlguidance) {
            logger.warn(
                "YALS was not built with LLGuidance. Skipping JSON schema.",
            );

            return;
        }

        const grammarArray = ["start: json_object"];
        const schemaString = JSON.stringify(
            schema,
            null,
            2,
        );
        grammarArray.push(`json_object: %json ${schemaString}`);

        this.sampler.llguidance(grammarArray.join("\n"));
    }

    regex(regex: string) {
        if (!hasLlguidance) {
            logger.warn(
                "YALS was not built with LLGuidance. Skipping Regex parsing.",
            );

            return;
        }

        const grammarArray = ["start: text"];
        grammarArray.push(`text: ${regex}`);

        this.sampler.llguidance(grammarArray.join("\n"));
    }
}
