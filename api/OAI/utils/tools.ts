import * as z from "@/common/myZod.ts";

import { ToolCall } from "../types/tools.ts";

export const TOOL_CALL_SCHEMA = {
    $schema: "http://json-schema.org/draft-07/schema#",
    type: "array",
    items: {
        type: "object",
        properties: {
            id: { type: "string" },
            function: {
                type: "object",
                properties: {
                    name: { type: "string" },
                    arguments: {
                        // Converted to OAI's string in post process
                        type: "object",
                    },
                },
                required: ["name", "arguments"],
            },
            type: { type: "string", enum: ["function"] },
        },
        required: ["id", "function", "type"],
    },
};

export class ToolCallProcessor {
    static fromJson(toolCallsString: string) {
        const rawToolObject = JSON.parse(toolCallsString);

        // Parse from deserialized object
        const ToolListSchema = z.array(ToolCall);
        const toolCalls = ToolListSchema.parse(rawToolObject);
        const updatedToolCalls = toolCalls.map((toolCall) => {
            toolCall.function.arguments = JSON.stringify(
                toolCall.function.arguments,
            );
            return toolCall;
        });

        return updatedToolCalls;
    }

    static toJson() {
    }

    static dump() {
    }
}
