import ScreenSaver
import AppKit
import QuartzCore
import Metal

@objc(CatScreensaverView)
public class CatScreensaverView: ScreenSaverView {

    private var renderer: MetalRenderer?
    private var metalLayer: CAMetalLayer?
    private var rendererInitFailed = false

    public override init?(frame: NSRect, isPreview: Bool) {
        super.init(frame: frame, isPreview: isPreview)
        setupView()
    }

    public required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        NSLog("CAT3D: setupView begin (bounds=\(bounds), isPreview=\(isPreview))")
        animationTimeInterval = 1.0 / 60.0

        // Use the standard "Metal as a sublayer" pattern. We do NOT override
        // makeBackingLayer() — under legacyScreenSaver / WallpaperAgent the
        // host expects a normal CALayer at the root and may otherwise wrap or
        // replace a custom backing layer, leaving us with a stale reference.
        wantsLayer = true
        let root: CALayer
        if let existing = self.layer {
            root = existing
        } else {
            let l = CALayer()
            self.layer = l
            root = l
        }
        root.backgroundColor = NSColor.black.cgColor

        let metal = CAMetalLayer()
        metal.pixelFormat = .bgra8Unorm
        metal.framebufferOnly = true
        metal.isOpaque = true
        metal.needsDisplayOnBoundsChange = true
        metal.contentsGravity = .resize
        metal.frame = bounds
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        metal.contentsScale = scale
        metal.drawableSize = CGSize(width:  max(bounds.width  * scale, 1),
                                     height: max(bounds.height * scale, 1))
        root.addSublayer(metal)
        self.metalLayer = metal

        NSLog("CAT3D: metal sublayer attached (frame=\(metal.frame), drawableSize=\(metal.drawableSize), scale=\(scale))")
    }

    /// Build the renderer the first time we actually have a non-zero drawable.
    private func ensureRendererReady() {
        guard renderer == nil, !rendererInitFailed, let metal = metalLayer else { return }
        guard metal.drawableSize.width >= 1, metal.drawableSize.height >= 1 else { return }
        do {
            renderer = try MetalRenderer(layer: metal)
            NSLog("CAT3D: MetalRenderer instantiated successfully")
        } catch {
            rendererInitFailed = true
            NSLog("CAT3D: renderer init failed: \(error)")
        }
    }

    public override func layout() {
        super.layout()
        guard let metal = metalLayer else { return }
        let scale = window?.backingScaleFactor ?? metal.contentsScale
        metal.contentsScale = scale
        // Disable implicit animation so resizes don't lag behind the view.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        metal.frame = bounds
        metal.drawableSize = CGSize(width:  max(bounds.width  * scale, 1),
                                     height: max(bounds.height * scale, 1))
        CATransaction.commit()
    }

    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // Window is now real — pick up its backing scale and drawable size.
        layout()
        ensureRendererReady()
    }

    public override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        layout()
    }

    public override func animateOneFrame() {
        super.animateOneFrame()
        ensureRendererReady()
        renderer?.render()
    }

    public override var hasConfigureSheet: Bool { false }
    public override var configureSheet: NSWindow? { nil }
}
