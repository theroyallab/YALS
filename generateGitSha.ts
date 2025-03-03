import { getCommitSha } from "@/common/utils.ts";

if (import.meta.main) {
    const sha = await getCommitSha();

    if (sha) {
        await Deno.writeTextFile("gitSha.txt", sha);
        console.log(`Successfully wrote Git SHA (${sha}) to gitSha.txt.`);
    } else {
        console.log("Failed to write Git SHA due to the errors above.");
    }
}
