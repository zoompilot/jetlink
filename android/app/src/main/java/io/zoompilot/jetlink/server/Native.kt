package io.zoompilot.jetlink.server

/**
 * libjetlink.so: the jetlink server in Swift, the same code the iPhone and
 * Mac apps run (JetlinkKit/Sources/JetlinkAndroid/Bridge.swift). Strings in
 * and out, JSON where there is structure. The calls that wait (command,
 * snapshot, logs) belong on a background thread.
 */
object Native {
    init {
        System.loadLibrary("jetlink")
    }

    /** Starts the server; null, or why it could not start. */
    external fun start(config: String): String?

    /** Stops the server and releases the engine. */
    external fun stop()

    /** One control command, `{"cmd": "prepare", ...}`; returns its reply. Blocks. */
    external fun command(command: String): String

    /** The app's state once its version passes [after], or after [timeoutMs]; empty when nothing changed. Blocks. */
    external fun snapshot(after: Long, timeoutMs: Int): String

    /** Log lines numbered after [after]: `{"next": n, "lines": [...]}`. */
    external fun logs(after: Long): String

    /**
     * The comma's gadget, opened with its vendor interface claimed: the
     * connection's file descriptor and the bulk endpoints' addresses. Null
     * or an error.
     */
    external fun usbAttach(fd: Int, inEndpoint: Int, outEndpoint: Int): String?

    /**
     * One of the comma's add-on interfaces on the same connection, claimed
     * too: its class triple and its bulk endpoints' addresses. Null or an error.
     */
    external fun usbAttachAddon(fd: Int, kind: Int, subclass: Int, protocol: Int, inEndpoint: Int, outEndpoint: Int): String?

    /** The gadget is going, add-ons and all; returns once nothing uses the descriptor. */
    external fun usbDetach()

    /** "nominal", "fair", "serious" or "critical", for the benchmark's reports. */
    external fun reportThermal(label: String)

    /** The onnxruntime version, for Settings before the server has said it. */
    external fun runtimeVersion(): String
}
