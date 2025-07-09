import * as z from "@/common/myZod.ts";

const Function = z.object({
    name: z.string(),
    description: z.string(),
    parameters: z.record(z.string(), z.unknown()),
});

export const ToolSpec = z.object({
    function: Function,
    type: z.literal("function"),
});

export type ToolSpec = z.infer<typeof ToolSpec>;

const Tool = z.object({
    name: z.string(),
    arguments: z.union([z.string(), z.record(z.string(), z.unknown())]),
});

export const ToolCall = z.object({
    id: z.string().default(() => {
        return crypto.randomUUID().replaceAll("-", "").substring(0, 9);
    }),
    function: Tool,
    type: z.literal("function").default("function"),
});

export type ToolCall = z.infer<typeof ToolCall>;
