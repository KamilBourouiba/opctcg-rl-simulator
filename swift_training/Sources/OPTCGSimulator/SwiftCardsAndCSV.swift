import Foundation

/// Carte minimale (CSV export tcgcsv ou équivalent).
public struct SwiftCard: Sendable {
    public var cardId: String
    public var name: String
    public var cost: Int
    public var power: Int
    public var counter: Int
    public var color: String
    public var cardText: String
    public var cardType: String
    public var life: Int
    public var donDeckSize: Int
    public var hasActivateMain: Bool

    public init(
        cardId: String,
        name: String,
        cost: Int,
        power: Int,
        counter: Int,
        color: String,
        cardText: String,
        cardType: String,
        life: Int = 0,
        donDeckSize: Int = 10,
        hasActivateMain: Bool = false,
    ) {
        self.cardId = cardId
        self.name = name
        self.cost = cost
        self.power = power
        self.counter = counter
        self.color = color
        self.cardText = cardText
        self.cardType = cardType
        self.life = life
        self.donDeckSize = donDeckSize
        self.hasActivateMain = hasActivateMain
    }

    /// Vecteur déterministe (Swift natif — pas le même hash que Python ``hash()`` session-local).
    public func embedding(dim: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        var h: UInt64 = 1469598103934665603
        for b in cardId.utf8 {
            h ^= UInt64(b)
            h &*= 1099511628211
        }
        let base = Int64(bitPattern: h)
        let span = max(1, 256 * dim)
        let bb = abs(base % Int64(span))
        for i in 0..<dim {
            let shift = (i * 3) % 56
            let byte = Int((bb >> shift) & 255)
            v[i] = Float(byte) / 255.0
        }
        return v
    }

    public var loweredText: String { cardText.lowercased() }
    public var loweredType: String { cardType.lowercased() }
}

public enum SwiftDeckIO {
    public static func parseDeckFile(at url: URL) throws -> [(String, Int)] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [(String, Int)] = []
        let lineRe = try NSRegularExpression(pattern: #"^\s*(\d+)\s*[xX]\s*([A-Za-z0-9\-]+)\s*$"#)
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw).trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let ns = line as NSString
            guard let m = lineRe.firstMatch(in: line, range: NSRange(location: 0, length: ns.length)),
                  m.numberOfRanges >= 3
            else { continue }
            let n = Int(ns.substring(with: m.range(at: 1))) ?? 0
            let cid = ns.substring(with: m.range(at: 2)).uppercased()
            out.append((cid, n))
        }
        return out
    }

    public static func multiset(_ entries: [(String, Int)]) -> [String: Int] {
        var m: [String: Int] = [:]
        for (cid, n) in entries {
            m[cid, default: 0] += n
        }
        return m
    }

    public static func flatDeck(_ m: [String: Int]) -> [String] {
        var deck: [String] = []
        for cid in m.keys.sorted() {
            guard let n = m[cid] else { continue }
            deck.append(contentsOf: repeatElement(cid, count: n))
        }
        return deck
    }

    public static func readLeaderDirective(at url: URL) throws -> String? {
        let text = try String(contentsOf: url, encoding: .utf8)
        let leaderRe = try NSRegularExpression(
            pattern: #"^\s*(?:#\s*)?(?:leader|LEADER)\s*[:\s]+\s*([A-Za-z0-9\-]+)\s*$"#,
            options: [.caseInsensitive],
        )
        let deckTypeRe = try NSRegularExpression(
            pattern: #"deck_type\s*:\s*([A-Za-z0-9\-]+)"#,
            options: [.caseInsensitive],
        )
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw).trimmingCharacters(in: .whitespaces)
            let ns = line as NSString
            guard let m = leaderRe.firstMatch(in: line, range: NSRange(location: 0, length: ns.length))
            else { continue }
            return ns.substring(with: m.range(at: 1)).uppercased()
        }
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw)
            let ns = line as NSString
            guard let m = deckTypeRe.firstMatch(in: line, range: NSRange(location: 0, length: ns.length))
            else { continue }
            return ns.substring(with: m.range(at: 1)).uppercased()
        }
        return nil
    }

    public static func extractLeader(
        multiset m: inout [String: Int],
        cards: [String: SwiftCard],
        explicitLeaderId: String?,
        preferredOrder: [String],
    ) -> String? {
        func isLeader(_ cid: String) -> Bool {
            guard let c = cards[cid] else { return false }
            return c.loweredType.trimmingCharacters(in: .whitespacesAndNewlines) == "leader"
        }
        var chosen: String?
        if let e = explicitLeaderId?.uppercased(), (m[e] ?? 0) >= 1, isLeader(e) {
            chosen = e
        }
        if chosen == nil {
            for cid in preferredOrder where (m[cid] ?? 0) >= 1 && isLeader(cid) {
                chosen = cid
                break
            }
        }
        if chosen == nil {
            for cid in m.keys.sorted() where (m[cid] ?? 0) >= 1 && isLeader(cid) {
                chosen = cid
                break
            }
        }
        guard let lid = chosen else { return nil }
        // Python parity: leader is out of the 50-card deck, remove all copies from multiset.
        m.removeValue(forKey: lid)
        return lid
    }
}

public enum SwiftCSV {
    public static func loadCards(path: URL, columnMap: [String: String]) throws -> [String: SwiftCard] {
        let text = try String(contentsOf: path, encoding: .utf8)
        let lines = text.split(whereSeparator: \.isNewline).filter { !$0.isEmpty }
        guard let headerLine = lines.first else { return [:] }
        let headers = splitCsvRow(String(headerLine))

        func col(_ logical: String, fallbacks: [String]) -> Int? {
            if let map = columnMap[logical], let i = headers.firstIndex(of: map) { return i }
            for f in fallbacks {
                if let i = headers.firstIndex(of: f) { return i }
            }
            return nil
        }

        guard let idIdx = col("card_id", fallbacks: ["card_id", "Number", "number", "productId", "id"]) else {
            return [:]
        }
        let nameIdx = col("name", fallbacks: ["name", "Name", "cleanName"]) ?? idIdx
        let costIdx = col("cost", fallbacks: ["cost", "Cost"])
        let powIdx = col("power", fallbacks: ["power", "Power"])
        let ctrIdx = col("counter", fallbacks: ["counter", "Counter"])
        let colorIdx = col("color", fallbacks: ["color", "Color"])
        let txtIdx = col("card_text", fallbacks: ["card_text", "Description", "description"])
        let typeIdx = col("card_type", fallbacks: ["card_type", "Type", "type"])
        let lifeIdx = col("life", fallbacks: ["life", "Life"])
        let donDeckIdx = col("don_deck_size", fallbacks: ["don_deck_size", "Don Deck Size", "don_deck"])

        var db: [String: SwiftCard] = [:]
        for line in lines.dropFirst() {
            let cols = splitCsvRow(String(line))
            guard idIdx < cols.count else { continue }
            let cid = cols[idIdx].trimmingCharacters(in: .whitespaces)
            if cid.isEmpty { continue }
            let name = nameIdx < cols.count ? cols[nameIdx] : cid
            let cost = costIdx.map { Int(cols[$0]) ?? 0 } ?? 0
            let pow = powIdx.map { Int(cols[$0]) ?? 0 } ?? 0
            let ctr = ctrIdx.map { Int(cols[$0]) ?? 0 } ?? 0
            let color = colorIdx.map { cols[$0] } ?? ""
            let txt = txtIdx.map { cols[$0] } ?? ""
            let ctype = typeIdx.map { cols[$0] } ?? ""
            let life = lifeIdx.map { Int(cols[$0]) ?? 0 } ?? 0
            let donDeck = donDeckIdx.map { Int(cols[$0]) ?? 10 } ?? 10
            let lt = txt.lowercased()
            let hasAM = lt.contains("[activate: main]")
            let card = SwiftCard(
                cardId: cid,
                name: name,
                cost: cost,
                power: pow,
                counter: ctr,
                color: color,
                cardText: txt,
                cardType: ctype,
                life: life,
                donDeckSize: donDeck,
                hasActivateMain: hasAM,
            )
            db[cid] = card
        }
        return db
    }

    private static func splitCsvRow(_ line: String) -> [String] {
        var out: [String] = []
        out.reserveCapacity(32)
        var cell = ""
        var inQuotes = false
        let chars = Array(line)
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if ch == "\"" {
                if inQuotes, i + 1 < chars.count, chars[i + 1] == "\"" {
                    cell.append("\"")
                    i += 2
                    continue
                }
                inQuotes.toggle()
                i += 1
                continue
            }
            if ch == ",", !inQuotes {
                out.append(cell.trimmingCharacters(in: .whitespaces))
                cell.removeAll(keepingCapacity: true)
                i += 1
                continue
            }
            cell.append(ch)
            i += 1
        }
        out.append(cell.trimmingCharacters(in: .whitespaces))
        return out
    }
}
