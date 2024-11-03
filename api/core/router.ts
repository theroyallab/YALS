import { Hono } from "hono";

const router = new Hono();

router.get(
    "/hello",
    (c) => {
        return c.text("Hello");
    },
);

export default router;
