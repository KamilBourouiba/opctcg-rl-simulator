import Foundation

/// Extraction de segments d'effet par timing (équivalent ``extract_timing_segment`` / ``_clip_timing_segment_body`` en Python).
enum TextTimingSegments {
    static let maxSegmentChars = 16_384

    /// Même normalisation que ``_norm`` (``on_play_resolver.py``).
    static func norm(_ s: String) -> String {
        s
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
    }

    static func clipTimingSegmentBody(_ rest: String, maxChars: Int = maxSegmentChars) -> String {
        var rest = rest.trimmingCharacters(in: .whitespacesAndNewlines)
        if rest.isEmpty { return rest }
        if let m = reTimingStrictNewline.firstMatch(
            in: rest,
            options: [],
            range: NSRange(location: 0, length: (rest as NSString).length)
        ) {
            rest = (rest as NSString).substring(to: m.range.location)
        } else if let m = reAnyBracketNewline.firstMatch(
            in: rest,
            options: [],
            range: NSRange(location: 0, length: (rest as NSString).length)
        ) {
            rest = (rest as NSString).substring(to: m.range.location)
        }
        if let m = reTimingInline.firstMatch(
            in: rest,
            options: [],
            range: NSRange(location: 0, length: (rest as NSString).length)
        ) {
            rest = (rest as NSString).substring(to: m.range.location)
        }
        if rest.count > maxChars {
            let end = rest.index(rest.startIndex, offsetBy: maxChars)
            rest = String(rest[..<end])
        }
        return rest.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Effet [On Play] : segment après le label, ou premier « paragraphe » ``clip`` sur la carte (héritage Python).
    static func extractOnPlayEffectWindow(_ cardText: String) -> String {
        let t = norm(cardText)
        if let m = reOnPlayBracket.firstMatch(
            in: t,
            options: [],
            range: NSRange(location: 0, length: (t as NSString).length)
        ) {
            let from = m.range.location + m.range.length
            let rest = (t as NSString).substring(from: from)
            return clipTimingSegmentBody(rest)
        }
        if let m = reOnPlayLine.firstMatch(
            in: t,
            options: [],
            range: NSRange(location: 0, length: (t as NSString).length)
        ) {
            let from = m.range.location + m.range.length
            let rest = (t as NSString).substring(from: from)
            return clipTimingSegmentBody(rest)
        }
        return clipTimingSegmentBody(t)
    }

    /// Après un label de timing (sauf on_play / activate_main). ``nil`` si le label est absent.
    static func extractBodyAfterLabel(_ cardText: String, timing: String) -> String? {
        let t = norm(cardText)
        let re: NSRegularExpression
        switch timing {
        case "main":
            re = reLabelMain
        case "trigger":
            re = reLabelTrigger
        case "on_ko":
            re = reLabelOnKO
        case "when_attacking":
            re = reLabelWhenAttacking
        case "end_of_your_turn":
            re = reLabelEndOfYourTurn
        default:
            return nil
        }
        guard let m = re.firstMatch(
            in: t,
            options: [],
            range: NSRange(location: 0, length: (t as NSString).length)
        ) else { return nil }
        let from = m.range.location + m.range.length
        let rest = (t as NSString).substring(from: from)
        let clipped = clipTimingSegmentBody(rest)
        return clipped.isEmpty ? nil : clipped
    }

    /// [Activate: Main] — texte d'effet après le coût (``:``), aligné sur ``apply_activate_main_effects``.
    static func extractActivateMainEffectText(_ cardText: String) -> String? {
        let t = norm(cardText)
        let ns = t as NSString
        let len = ns.length
        guard let m = reLabelActivateMain.firstMatch(
            in: t,
            options: [],
            range: NSRange(location: 0, length: len)
        ) else { return nil }
        let rest = ns.substring(from: m.range.location + m.range.length)
        var effect: String
        if let c = reActivateColonParens.firstMatch(
            in: rest,
            options: [],
            range: NSRange(location: 0, length: (rest as NSString).length)
        ) {
            effect = (rest as NSString).substring(from: c.range.location + c.range.length)
        } else if let c2 = reActivateDonOrOptColon.firstMatch(
            in: rest,
            options: [],
            range: NSRange(location: 0, length: (rest as NSString).length)
        ) {
            effect = (rest as NSString).substring(from: c2.range.location + c2.range.length)
        } else {
            effect = rest
        }
        let clipped = clipTimingSegmentBody(effect.trimmingCharacters(in: .whitespacesAndNewlines))
        return clipped.isEmpty ? nil : clipped
    }

    /// Dernier ``[DON!! xN]`` dans une fenêtre (comme l’énumération Python sur le préfixe).
    private static func donXMaxInWindow(_ window: String) -> Int {
        var best = 0
        reDonXBlock.enumerateMatches(
            in: window,
            options: [],
            range: NSRange(location: 0, length: (window as NSString).length)
        ) { m2, _, _ in
            if let m2, m2.numberOfRanges >= 2 {
                let sub = (window as NSString).substring(with: m2.range(at: 1))
                if let v = Int(sub) { best = v }
            }
        }
        return best
    }

    /// ``[DON!! xN]`` s'applique au bloc s'il apparaît juste avant l'en-tête de timing.
    /// - Note: pour ``on_play``, si aucun marqueur (héritage : tout le texte clippé), on lit les ``[DON!! xN]`` sur les 300 premiers caractères (comme un préfixe avant le « faux » label en 0).
    static func donXRequirementBeforeTimingLabel(_ cardText: String, timing: String) -> Int {
        let t = norm(cardText)
        let nsT = t as NSString
        let tLen = nsT.length
        let start: Int?
        switch timing {
        case "on_play":
            let fullR = NSRange(location: 0, length: tLen)
            if let m = reOnPlayBracket.firstMatch(in: t, options: [], range: fullR) {
                start = m.range.location
            } else if let m = reOnPlayLine.firstMatch(in: t, options: [], range: fullR) {
                start = m.range.location
            } else {
                start = nil
            }
        case "when_attacking":
            let re = reLabelWhenAttacking
            let m = re.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        case "on_ko":
            let m = reLabelOnKO.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        case "main":
            let m = reLabelMain.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        case "trigger":
            let m = reLabelTrigger.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        case "activate_main":
            let m = reLabelActivateMain.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        case "end_of_your_turn":
            let m = reLabelEndOfYourTurn.firstMatch(
                in: t,
                options: [],
                range: NSRange(location: 0, length: (t as NSString).length)
            )
            start = m.map { $0.range.location }
        default:
            start = nil
        }
        if timing == "on_play", start == nil {
            let preLen = min(300, tLen)
            let window = nsT.substring(with: NSRange(location: 0, length: preLen))
            return donXMaxInWindow(window)
        }
        guard let s0 = start else { return 0 }
        let lookback = min(300, s0)
        let f = s0 - lookback
        let window = nsT.substring(with: NSRange(location: f, length: lookback))
        return donXMaxInWindow(window)
    }
}

// MARK: - Regex (alignés sur ``on_play_resolver``)

private let reTimingStrictNewline: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\n\s*\[(?:"#
            + #"trigger|main|on\s*play|on\s*k\.?o\.?|when\s+attacking|"# +
            "activate\\s*:\\s*main|activate\\s*:\\s*counter|activate\\s*:\\s*action|" +
            "end\\s+of\\s+your\\s+turn|end\\s+of\\s+your\\s+opponent'?s?\\s+turn|" +
            "drip|once\\s+per\\s+turn|on\\s+your\\s+opponent'?s?\\s+attack|on\\s+block" +
            #")[^\]\n]*\]"#,
        options: .caseInsensitive,
    )
}()

private let reAnyBracketNewline: NSRegularExpression = {
    try! NSRegularExpression(pattern: #"\n\s*\[[^\n]{1,200}\]"#, options: .caseInsensitive)
}()

private let reTimingInline: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"(?<=[\w\)\]\.\"'])\s+\[(?:"# +
            #"trigger|main|on\s*play|on\s*k\.?o\.?|when\s+attacking|"# +
            "activate\\s*:\\s*main|activate\\s*:\\s*counter|end\\s+of\\s+your\\s+turn|drip|once\\s+per\\s+turn|on\\s+"

            + "your\\s+opponent'?s?\\s+attack" +
            #")[^\]\n]*\]"#,
        options: .caseInsensitive,
    )
}()

private let reOnPlayBracket: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[on\s*play\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()

private let reOnPlayLine: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"(?m)(?:^|\n)\s*on\s*play\s*:\s*"#,
        options: .caseInsensitive,
    )
}()

private let reLabelMain: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[main\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()
private let reLabelTrigger: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[trigger\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()
private let reLabelOnKO: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[on\s*k\.?o\.?\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()
private let reLabelWhenAttacking: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[when\s*attacking\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()
private let reLabelEndOfYourTurn: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[end\s*of\s*your\s*turn\]\s*:?\s*"#,
        options: .caseInsensitive,
    )
}()
private let reLabelActivateMain: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[activate\s*:\s*main\]"#,
        options: .caseInsensitive,
    )
}()
private let reActivateColonParens: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\)\s*:"#,
    )
}()
private let reDonXBlock: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"\[don!!\s*x\s*(\d+)\]"#,
        options: .caseInsensitive,
    )
}()
private let reActivateDonOrOptColon: NSRegularExpression = {
    try! NSRegularExpression(
        pattern: #"(?:don!!\s*-\s*\d+|once\s+per\s+turn)[^:]*:"#,
        options: .caseInsensitive,
    )
}()
