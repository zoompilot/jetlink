import Foundation

/// What a model takes on disk where onnxruntime prepares it for CoreML (a
/// Mac, an iPhone or an iPad), from the model's size alone. The shares are
/// an iPad Air's (M1) with onnxruntime 1.29.0, the same for Light Speed 2
/// (769 MB) and Lebowski (1.76 GB).
///
/// Too little room fails twice over. A build that runs out stops part way,
/// in CoreML's words "MIL FileWriter: Unknown error occured while writing
/// data to the file". And on a nearly full disk the system purges apps'
/// caches, the Neural Engine's compiled copies with them, and the Neural
/// Engine's compiler then has no room for new ones: CoreML runs every model
/// on the CPU, in every app, until space is freed. On that iPad Light Speed 2
/// went from 25 ms a frame to about 480.
public struct StorageNeed: Sendable, Equatable {
  /// What the space is for.
  public enum Step: Sendable, Equatable {
    /// To download the model and prepare it.
    case download
    /// To prepare a model already on disk.
    case prepare
    /// To keep a prepared model on the Neural Engine.
    case run
  }

  /// The model file.
  public let model: Int64
  /// The prepared copy kept after a build: the prepared ONNX and CoreML's
  /// compiled model, 1.9 times the model on both.
  public let prepared: Int64
  /// What a build writes on top and deletes when it is done: CoreML's
  /// converted model, about the model's size.
  public let buildScratch: Int64
  /// The Neural Engine's compiled copy, about the model's size. CoreML keeps
  /// it in the app's caches, which the system may purge when short, so the
  /// room for it is kept even once it exists.
  public let neuralEngine: Int64
  /// Room left for the system, so it has no cause to purge the copies above.
  public let headroom: Int64

  /// The room left for the system: on the iPad above, the Neural Engine
  /// failed twice with the disk nearly full and came back once a few GB were
  /// freed.
  public static let headroomBytes: Int64 = 3_000_000_000

  public init(model: Int64, prepared: Int64, buildScratch: Int64, neuralEngine: Int64, headroom: Int64) {
    self.model = model
    self.prepared = prepared
    self.buildScratch = buildScratch
    self.neuralEngine = neuralEngine
    self.headroom = headroom
  }

  /// A model of `modelBytes` prepared for CoreML.
  public static func coreML(modelBytes: Int64) -> StorageNeed {
    let bytes = max(0, modelBytes)
    return StorageNeed(model: bytes, prepared: bytes * 2, buildScratch: bytes, neuralEngine: bytes, headroom: headroomBytes)
  }

  /// The free space a step needs.
  public func bytes(for step: Step) -> Int64 {
    switch step {
    case .download: model + prepared + buildScratch + neuralEngine + headroom
    case .prepare: prepared + buildScratch + neuralEngine + headroom
    case .run: neuralEngine + headroom
    }
  }

  /// Free space to prepare the model for `profiles` more processors in
  /// turn, as Compare Processors does: a prepared copy each, a Neural Engine
  /// copy for each of `onNeuralEngine` of them, and one build's working copy,
  /// since each goes when its build ends.
  public func toPrepare(profiles: Int, onNeuralEngine: Int) -> Int64 {
    guard profiles > 0 else { return 0 }
    return Int64(profiles) * prepared + Int64(onNeuralEngine) * neuralEngine + buildScratch + headroom
  }

  /// What is missing for a step with `free` bytes free, or nil when they are enough.
  public func shortfall(for step: Step, free: Int64, name: String? = nil) -> StorageShortfall? {
    let need = bytes(for: step)
    guard free < need else { return nil }
    return StorageShortfall(step: step, need: need, free: free, message: message(for: step, need: need, free: free, name: name))
  }

  private func message(for step: Step, need: Int64, free: Int64, name: String?) -> String {
    let model = name.map { "\($0) needs" } ?? "This model needs"
    let have = "\(Self.gigabytes(free)) is free"
    switch step {
    case .download:
      return "\(model) about \(Self.gigabytes(need)) free to download and prepare: the model \(Self.gigabytes(self.model)), "
        + "its prepared copy \(Self.gigabytes(prepared)), CoreML's working copy \(Self.gigabytes(buildScratch)), "
        + "the Neural Engine's copy \(Self.gigabytes(neuralEngine)) and \(Self.gigabytes(headroom)) left for the system. "
        + "\(have). Free some space and try again."
    case .prepare:
      return "\(model) about \(Self.gigabytes(need)) free to prepare: its prepared copy \(Self.gigabytes(prepared)), "
        + "CoreML's working copy \(Self.gigabytes(buildScratch)), the Neural Engine's copy \(Self.gigabytes(neuralEngine)) "
        + "and \(Self.gigabytes(headroom)) left for the system. \(have). Free some space and try again."
    case .run:
      return "\(model) about \(Self.gigabytes(need)) free to keep running on the Neural Engine, and \(have). "
        + "With less, the system clears the Neural Engine's copy of the model and it runs on the CPU, far too slow to drive."
    }
  }

  /// Bytes as the messages show them, in decimal GB like the system's own storage screens.
  public static func gigabytes(_ bytes: Int64) -> String {
    String(format: "%.1f GB", Double(bytes) / 1e9)
  }
}

/// A step a disk has too little room for.
public struct StorageShortfall: Sendable, Equatable {
  public let step: StorageNeed.Step
  /// The free space the step needs.
  public let need: Int64
  /// The free space there is.
  public let free: Int64
  /// One paragraph for the person: what is needed, what there is, what to do.
  public let message: String
}

/// Frames served far over the budget: on an iPhone or iPad, a model the
/// Neural Engine is not running. CoreML runs it on the CPU instead, at about
/// 400 ms a frame for Light Speed 2 on an iPad Air (M1) against 25, and the
/// comma keeps to its small model. A full disk is the usual cause: the Neural
/// Engine had no room for its copy of the model.
public enum SlowFrames {
  /// The comma's frame budget.
  public static let budgetMs = 50.0
  /// A mean of more than this, the budget twice over, is not a busy moment.
  public static let thresholdMs = 2 * budgetMs

  /// What to tell the person about frames averaging `meanMs`, with what the
  /// model in use is short of on disk, if anything; nil while frames are near
  /// the budget.
  public static func warning(meanMs: Double, frames: Int, storage: StorageShortfall?) -> String? {
    guard frames > 0, meanMs > thresholdMs else { return nil }
    let times = Int((meanMs / budgetMs).rounded())
    let head =
      "Frames take \(Int(meanMs.rounded())) ms, \(times) times the \(Int(budgetMs)) ms budget, "
      + "so the comma uses its small model. The Neural Engine is not running this model."
    if let storage {
      return head + " The likely cause is storage: " + storage.message
    }
    return head + " This usually means too little free space for the Neural Engine's copy of the model. "
      + "Free some space, or restart the device if there is plenty."
  }
}
