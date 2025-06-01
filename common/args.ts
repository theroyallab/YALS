// @ts-types="npm:@types/command-line-args"
import commandLineArgs from "command-line-args";

// @ts-types="npm:@types/command-line-usage";
import commandLineUsage from "command-line-usage";
import * as z from "@/common/myZod.ts";
import { ConfigSchema } from "@/common/configModels.ts";

// Replicates Python's strtobool for handling boolean values
function strToBool(value: string): boolean {
    return z.stringbool().parse(value);
}

// Converts the ConfigSchema to CLI arguments
function configToArgs() {
    const configGroups: commandLineUsage.OptionList[] = [];

    // Iterate and create groups from top-level arguments
    for (const [groupName, params] of Object.entries(ConfigSchema.shape)) {
        const groupOptions = createGroupOptions(groupName, params.shape);
        configGroups.push({ header: groupName, optionList: groupOptions });
    }

    return configGroups;
}

// Creates inner arg options for argument groups
function createGroupOptions(groupName: string, shape: z.ZodRawShape) {
    return Object.entries(shape).map(([key, value]) => {
        const option: commandLineUsage.OptionDefinition = {
            name: key.replaceAll("_", "-"),
            group: groupName,
        };

        setArgType(option, value);
        return option;
    });
}

// Converts a Zod schema type to a command-line-args type
// Drills down recursively until primitive types are found
function setArgType(
    option: commandLineUsage.OptionDefinition,
    zodType: z.core.$ZodType,
) {
    // Use _zod for fetching the underlying type
    const typeName = zodType._zod.def.type;

    switch (typeName) {
        case "string":
            option["type"] = String;
            break;
        case "number":
            option["type"] = Number;
            break;
        case "boolean":
            option["type"] = strToBool;
            break;
        case "optional":
            setArgType(
                option,
                (zodType as z.ZodOptional<z.ZodType>).unwrap(),
            );
            break;
        case "nullable":
            setArgType(
                option,
                (zodType as z.ZodNullable<z.ZodType>).unwrap(),
            );
            break;
        case "union":
            setArgType(
                option,
                (zodType as z.ZodUnion<[z.ZodType, ...z.ZodType[]]>).def
                    .options[0],
            );
            break;
        case "pipe":
            setArgType(
                option,
                (zodType as z.ZodPipe<z.ZodType>).def.in,
            );
            break;
        case "array":
            option["multiple"] = true;
            setArgType(option, (zodType as z.ZodArray<z.ZodType>).element);
            break;
    }
}

// Parses global arguments from Deno.args
export function parseArgs() {
    // Define option groups
    const helpGroup: commandLineUsage.Section = {
        header: "Support",
        optionList: [{
            name: "help",
            type: Boolean,
            description: "Prints this menu",
            group: "support",
        }],
    };

    const epilog: commandLineUsage.Section = {
        header: "Epilog",
        content: "- strtobool flags require an explicit value. " +
            "Example: --flash-attention true",
        raw: true,
    };

    const configGroups = configToArgs();
    const optionGroups = [...configGroups, helpGroup];
    const usage = commandLineUsage([...optionGroups, epilog]);
    const cliOptions: commandLineUsage.OptionDefinition[] = optionGroups
        .flatMap((option) => option.optionList ?? []);

    // Parse the options
    const args = commandLineArgs(cliOptions, { argv: Deno.args });

    // Replace keys with underscores for config parsing
    for (const groupName of Object.keys(args)) {
        const groupArgs = args[groupName];

        if (groupArgs && typeof groupArgs === "object") {
            args[groupName] = Object.fromEntries(
                Object.entries(groupArgs).map((
                    [k, v],
                ) => [k.replaceAll("-", "_"), v]),
            );
        }
    }

    return { args, usage };
}
