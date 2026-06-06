# PokerStars Hand History Format Reference

## Complete Format Specification

### 1. Hand Header
```
PokerStars Hand #{hand_id}: Hold'em No Limit (${sb}/{bb} USD) - {YYYY/MM/DD HH:MM:SS ET}
```
- `hand_id`: 18-19 digit integer
- Stakes: `$0.05/$0.10 USD` format
- Timestamp: Eastern Time

### 2. Table Line
```
Table '{table_name}' {max_players}-max Seat #{button_seat} is the button
```
- Table name: `'WPT Global'`
- Max players: `8-max`
- Button seat: 1-8

### 3. Seat Declarations (all players)
```
Seat {N}: {player_name} (${stack} in chips)
```
- Seats numbered 1-8
- Stack in USD: `$50.60`

### 4. Antes (every player posts)
```
{player_name}: posts the ante ${ante}
```
- Listed in seat order
- Amount = total_antes / n_players (rounded to $0.05 or $0.10)

### 5. Blinds
```
{sb_player}: posts small blind ${sb_amount}
{bb_player}: posts big blind ${bb_amount}
```
- Standard: SB=$0.05, BB=$0.10

### 6. Straddle (optional, UTG or dealer button)
```
{str_player}: posts straddle ${str_amount}
```
- Straddle = 2× BB: `$0.20`

### 7. Hole Cards Section
```
*** HOLE CARDS ***
Dealt to {hero_name} [{card1} {card2}]
```
- Hero = the player recording (dLzinN typically)
- Cards: `Ad As`, `Qd Qh`, etc. (rank+suit, no brackets inside)
- If no hero visible: no `Dealt to` line

### 8. Preflop Actions
```
{player}: folds
{player}: calls ${amount}
{player}: calls ${amount} and is all-in
{player}: raises ${delta} to ${total}
{player}: raises ${delta} to ${total} and is all-in
{player}: checks
```

### 9. Flop Section
```
*** FLOP *** [{card1} {card2} {card3}]
{...actions...}
```

### 10. Turn Section
```
*** TURN *** [{flop_card1} {flop_card2} {flop_card3}] [{turn_card}]
{...actions...}
```

### 11. River Section
```
*** RIVER *** [{flop1} {flop2} {flop3} {turn}] [{river_card}]
{...actions...}
```

### 12. Showdown
```
*** SHOWDOWN ***
{player}: shows [{card1} {card2}] ({hand_description})
```
- Hand descriptions: `a full house, Sixes full of Eights`, `two pair, Aces and Eights`, etc.

### 13. Winner Collection
```
{winner}: collected ${amount} from pot
```
- One line per winner
- Amount = total pot in USD

### 14. Summary Section
```
*** SUMMARY ***
Total pot ${amount} | Rake $0
Board [{c1} {c2} {c3} {c4} {c5}]
Seat {N}: {name} folded before Flop (didn't bet)
Seat {N}: {name} folded before Flop
Seat {N}: {name} (small blind) folded before Flop (didn't bet)
Seat {N}: {name} (big blind) folded before Flop
Seat {N}: {name} (button) folded before Flop (didn't bet)
Seat {N}: {name} folded on the Flop
Seat {N}: {name} folded on the Turn
Seat {N}: {name} folded on the River
Seat {N}: {name} showed [{c1} {c2}] and won (${amount}) with {hand_desc}
Seat {N}: {name} showed [{c1} {c2}] and lost with {hand_desc}
Seat {N}: {name} collected (${amount})
```
- Board line omitted for preflop-only hands
- Role in parentheses: `(button)`, `(small blind)`, `(big blind)`
- `(didn't bet)` appended if player never voluntarily bet/raised

---

## Card Format
- Ranks: `A K Q J T 9 8 7 6 5 4 3 2`
- Suits: `h` hearts, `d` diamonds, `c` clubs, `s` spades
- Examples: `Ah`, `Kd`, `Qc`, `Js`, `Th`, `9d`

---

## Real Example (26-19.png ground truth)

```
PokerStars Hand #127929051220758184: Hold'em No Limit ($0.05/$0.10 USD) - 2026/05/26 12:31:54 ET
Table 'WPT Global' 8-max Seat #4 is the button
Seat 1: 奥德彪魇 ($83.35 in chips)
Seat 2: 淡淡的会顺顺 ($108.50 in chips)
Seat 3: dLzinN ($44.00 in chips)
Seat 4: 全场紧B型 ($15.65 in chips)
Seat 5: Kaakamelli ($5.54 in chips)
Seat 6: AlexanderN ($54.75 in chips)
Seat 7: FlushFever97 ($19.35 in chips)
奥德彪魇: posts the ante $0.05
淡淡的会顺顺: posts the ante $0.05
dLzinN: posts the ante $0.05
全场紧B型: posts the ante $0.05
Kaakamelli: posts the ante $0.05
AlexanderN: posts the ante $0.05
FlushFever97: posts the ante $0.05
Kaakamelli: posts small blind $0.05
AlexanderN: posts big blind $0.10
FlushFever97: posts straddle $0.20
*** HOLE CARDS ***
Dealt to dLzinN [Ad As]
奥德彪魇: folds
淡淡的会顺顺: raises $0.68 to $0.88
dLzinN: raises $2.01 to $2.89
全场紧B型: folds
Kaakamelli: folds
AlexanderN: folds
FlushFever97: folds
淡淡的会顺顺: calls $2.01
*** FLOP *** [8h 2c 6c]
淡淡的会顺顺: bets $3.24
dLzinN: raises $6.48 to $9.72
淡淡的会顺顺: calls $6.48
*** TURN *** [8h 2c 6c] [8c]
淡淡的会顺顺: checks
dLzinN: checks
*** RIVER *** [8h 2c 6c 8c] [9s]
淡淡的会顺顺: checks
dLzinN: bets $13.00
淡淡的会顺顺: raises $82.80 to $95.80 and is all-in
dLzinN: calls $18.34 and is all-in
*** SHOWDOWN ***
淡淡的会顺顺: shows [6h 6s] (a full house, Sixes full of Eights)
dLzinN: shows [Ad As] (two pair, Aces and Eights)
淡淡的会顺顺 collected $153.00 from pot
*** SUMMARY ***
Total pot $153.00 | Rake $0
Board [8h 2c 6c 8c 9s]
Seat 1: 奥德彪魇 folded before Flop (didn't bet)
Seat 2: 淡淡的会顺顺 showed [6h 6s] and won ($153.00) with a full house, Sixes full of Eights
Seat 3: dLzinN showed [Ad As] and lost with two pair, Aces and Eights
Seat 4: 全场紧B型 (button) folded before Flop (didn't bet)
Seat 5: Kaakamelli (small blind) folded before Flop
Seat 6: AlexanderN (big blind) folded before Flop
Seat 7: FlushFever97 folded before Flop
```

---

## Key Differences from Raw WPT Data

| WPT Screenshot | PokerStars Format |
|---|---|
| `Aumentar 6,80BB` | `raises $0.62 to $0.68` (delta + total) |
| `Fazer call 24,40BB` | `calls $2.44` |
| `Desistir` | `folds` |
| `Passar` | `checks` |
| `Aposta 20BB` | `bets $2.00` |
| Amounts in BB (European decimal: `,`) | Amounts in USD: `$0.10` |
| Player appears multiple times (multi-street) | One action block per street |
| Position badge: UTG, CO, BTN, SB, BB, STR | Role in summary only |
| Anonymous players (badge only, no name) | Must resolve name from position |

## Known OCR Limitations (unfixable)
- Hand ID: parser adds 1 extra digit (OCR artifact with long number)
- Chinese characters: some visually similar chars get swapped (e.g., 飚 ↔ 魇)
- All-in call amounts: sometimes OCR misses the exact stack amount
