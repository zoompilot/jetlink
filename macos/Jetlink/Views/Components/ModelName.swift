import Foundation

extension ModelRow {
  /// The name without the trailing parenthesised build date the catalog carries,
  /// so "BMRLNAP Model v4 (August 30, 2026)" reads as "BMRLNAP Model v4". A
  /// parenthesis that is not a date is part of the name and stays.
  var displayName: String {
    guard let date = name.range(of: ModelRow.trailingDatePattern, options: .regularExpression) else { return name }
    return String(name[name.startIndex..<date.lowerBound])
  }

  static let trailingDatePattern = " \\([A-Za-z]+ \\d{1,2}, \\d{4}\\)$"
}
