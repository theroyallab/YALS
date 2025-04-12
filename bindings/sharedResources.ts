import { lib } from "./lib.ts";
import { ReadbackBuffer } from "./readbackBuffer.ts";

export class SharedResourceBundle {
  public rawPtr: Deno.PointerValue;
  private readbackBufferPtr: Deno.PointerValue;
  public samplerPtr: Deno.PointerValue;

  public readbackBuffer: ReadbackBuffer;

  constructor() {
    this.rawPtr = lib.symbols.resource_bundle_make();
    if(!this.rawPtr) throw new Error("Could not allocate shared resource bundle.");
    const view = new Deno.UnsafePointerView(this.rawPtr);
    this.readbackBufferPtr = Deno.UnsafePointer.create(view.getBigUint64(0));
    this.samplerPtr = Deno.UnsafePointer.create(view.getBigUint64(8));

    this.readbackBuffer = new ReadbackBuffer(this.readbackBufferPtr);
  }

  close() {
    lib.symbols.resource_bundle_release(this.rawPtr);
  }
}