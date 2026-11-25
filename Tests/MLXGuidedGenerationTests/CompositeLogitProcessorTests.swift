// Copyright © 2025 Apple Inc.

import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

/// Adds the number of emitted tokens to every logit.
private struct EmitCountProcessor: LogitProcessor {
    var emitted = 0

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        logits + Float(emitted)
    }

    mutating func didSample(token: MLXArray) {}

    mutating func didEmit(token: Int) {
        emitted += 1
    }
}

/// Multiplies logits by a scalar.
private struct ScaleProcessor: LogitProcessor {
    let scale: Float

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        logits * scale
    }

    mutating func didSample(token: MLXArray) {}
}

// `CompositeLogitProcessor` is a deprecated alias for `ChainedLogitProcessor`.
@Suite
struct CompositeLogitProcessorTests {

    @available(*, deprecated)
    @Test
    func deprecatedNameMatchesChainedLogitProcessor() {
        var composite = CompositeLogitProcessor([
            EmitCountProcessor(),
            ScaleProcessor(scale: 2.0),
        ])
        var chained = ChainedLogitProcessor(processors: [
            EmitCountProcessor(),
            ScaleProcessor(scale: 2.0),
        ])

        composite.didEmit(token: 1)
        chained.didEmit(token: 1)

        // (original + 1 emitted token) * 2.0
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let values = composite.process(logits: input).asArray(Float.self)

        #expect(values == [4.0, 6.0, 8.0])
        #expect(values == chained.process(logits: input).asArray(Float.self))
    }
}
