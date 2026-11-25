// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Use `ChainedLogitProcessor` from MLXLMCommon instead.
@available(*, deprecated, renamed: "ChainedLogitProcessor")
public typealias CompositeLogitProcessor = ChainedLogitProcessor

extension ChainedLogitProcessor {
    /// Keeps `CompositeLogitProcessor([...])` call sites compiling.
    @available(*, deprecated, renamed: "init(processors:)")
    public init(_ processors: [any LogitProcessor]) {
        self.init(processors: processors)
    }
}
