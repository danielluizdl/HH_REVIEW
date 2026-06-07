"""Generate PokerStars-format hand history from structured hand data."""

POSITION_MAP = {
    'BTN': 'button', 'SB': 'small blind', 'BB': 'big blind',
    'UTG': 'UTG', 'UTG+1': 'UTG+1', 'UTG+2': 'UTG+2',
    'MP': 'MP', 'MP+1': 'MP+1', 'HJ': 'HJ', 'CO': 'CO',
}


def _fmt(amount) -> str:
    """Format a dollar amount like PokerStars: $0.05, $108.50"""
    return f"${float(amount):.2f}"


def format_hand_history(hand_data: dict) -> str:
    """
    Convert structured hand_data dict to PokerStars format string.

    Expected hand_data keys:
      hand_id, timestamp, stakes (sb, bb), table_name, max_players,
      button_seat, players (list of {seat, name, stack}),
      antes, blinds (list of {name, type, amount}),
      straddle (optional: {name, amount}),
      hero, hole_cards (list of card strings),
      preflop_actions, flop_cards, flop_actions,
      turn_card, turn_actions, river_card, river_actions,
      showdown (list of {name, cards, hand_desc}),
      winners (list of {name, amount, pot_desc}),
      total_pot, rake, summary_seats
    """
    lines = []

    # Header
    hand_id = hand_data.get('hand_id', '0')
    ts = hand_data.get('timestamp', '2026/01/01 00:00:00 ET')
    sb = _fmt(hand_data.get('stakes', {}).get('sb', 0.05))
    bb = _fmt(hand_data.get('stakes', {}).get('bb', 0.10))
    lines.append(
        f"PokerStars Hand #{hand_id}: Hold'em No Limit ({sb}/{bb} USD) - {ts}"
    )

    # Table line
    table = hand_data.get('table_name', 'WPT Global')
    max_p = hand_data.get('max_players', 8)
    btn = hand_data.get('button_seat', 1)
    lines.append(f"Table '{table}' {max_p}-max Seat #{btn} is the button")

    # Seats
    for p in hand_data.get('players', []):
        lines.append(f"Seat {p['seat']}: {p['name']} ({_fmt(p['stack'])} in chips)")

    # Antes
    for p in hand_data.get('players', []):
        ante = hand_data.get('ante', 0)
        if ante > 0:
            lines.append(f"{p['name']}: posts the ante {_fmt(ante)}")

    # Blinds
    for blind in hand_data.get('blinds', []):
        btype = blind.get('type', 'blind')
        if btype == 'small blind':
            lines.append(f"{blind['name']}: posts small blind {_fmt(blind['amount'])}")
        elif btype == 'big blind':
            lines.append(f"{blind['name']}: posts big blind {_fmt(blind['amount'])}")

    # Straddle
    straddle = hand_data.get('straddle')
    if straddle:
        lines.append(f"{straddle['name']}: posts straddle {_fmt(straddle['amount'])}")

    # Hole cards
    lines.append("*** HOLE CARDS ***")
    hero = hand_data.get('hero')
    hole_cards = hand_data.get('hole_cards', [])
    if hero and hole_cards:
        cards_str = ' '.join(hole_cards)
        lines.append(f"Dealt to {hero} [{cards_str}]")

    # Preflop actions
    for action in hand_data.get('preflop_actions', []):
        lines.append(_format_action(action))

    # Flop
    flop = hand_data.get('flop_cards', [])
    if flop:
        cards_str = ' '.join(flop)
        lines.append(f"*** FLOP *** [{cards_str}]")
        for action in hand_data.get('flop_actions', []):
            lines.append(_format_action(action))

    # Turn
    turn = hand_data.get('turn_card')
    if turn:
        flop_str = ' '.join(flop)
        lines.append(f"*** TURN *** [{flop_str}] [{turn}]")
        for action in hand_data.get('turn_actions', []):
            lines.append(_format_action(action))

    # River
    river = hand_data.get('river_card')
    if river:
        board_so_far = flop + ([turn] if turn else [])
        board_str = ' '.join(board_so_far)
        lines.append(f"*** RIVER *** [{board_str}] [{river}]")
        for action in hand_data.get('river_actions', []):
            lines.append(_format_action(action))

    # Showdown
    showdown = hand_data.get('showdown', [])
    if showdown:
        lines.append("*** SHOWDOWN ***")
        for entry in showdown:
            cards_str = ' '.join(entry.get('cards', []))
            desc = entry.get('hand_desc', '')
            lines.append(f"{entry['name']}: shows [{cards_str}] ({desc})")

    # Winners
    for winner in hand_data.get('winners', []):
        pot_desc = winner.get('pot_desc', 'pot')
        lines.append(f"{winner['name']} collected {_fmt(winner['amount'])} from {pot_desc}")

    # Summary
    total_pot = hand_data.get('total_pot', 0)
    rake = hand_data.get('rake', 0)
    lines.append("*** SUMMARY ***")
    rake_str = "$0" if rake == 0 else _fmt(rake)
    lines.append(f"Total pot {_fmt(total_pot)} | Rake {rake_str}")

    # Board in summary
    board = flop + ([turn] if turn else []) + ([river] if river else [])
    if board:
        lines.append(f"Board [{' '.join(board)}]")

    # Summary seats
    for seat_info in hand_data.get('summary_seats', []):
        lines.append(_format_summary_seat(seat_info))

    return '\n'.join(lines)


def _format_action(action: dict) -> str:
    """Format a single player action line."""
    name = action.get('name', '')
    act = action.get('action', '')
    amount = action.get('amount')
    total = action.get('total')
    extra = action.get('extra', '')

    if act == 'folds':
        return f"{name}: folds"
    elif act == 'checks':
        return f"{name}: checks"
    elif act == 'calls':
        return f"{name}: calls {_fmt(amount)}{' and is all-in' if action.get('allin') else ''}"
    elif act == 'bets':
        return f"{name}: bets {_fmt(amount)}{' and is all-in' if action.get('allin') else ''}"
    elif act == 'raises':
        return f"{name}: raises {_fmt(amount)} to {_fmt(total)}{' and is all-in' if action.get('allin') else ''}"
    else:
        return f"{name}: {act}" + (f" {_fmt(amount)}" if amount else "")


def _format_summary_seat(seat_info: dict) -> str:
    """Format a summary seat line."""
    seat = seat_info.get('seat', '')
    name = seat_info.get('name', '')
    role = seat_info.get('role', '')
    outcome = seat_info.get('outcome', '')
    cards = seat_info.get('cards', [])
    amount = seat_info.get('amount')
    hand_desc = seat_info.get('hand_desc', '')

    role_str = f" ({role})" if role else ""

    if outcome == 'showed_won':
        if cards:
            cards_str = ' '.join(cards)
            return f"Seat {seat}: {name}{role_str} showed [{cards_str}] and won ({_fmt(amount)}) with {hand_desc}"
        else:
            return f"Seat {seat}: {name}{role_str} collected ({_fmt(amount)})"
    elif outcome == 'showed_lost':
        if cards:
            cards_str = ' '.join(cards)
            return f"Seat {seat}: {name}{role_str} showed [{cards_str}] and lost with {hand_desc}"
        else:
            return f"Seat {seat}: {name}{role_str} mucked hand"
    elif outcome == 'folded_preflop_no_bet':
        return f"Seat {seat}: {name}{role_str} folded before Flop (didn't bet)"
    elif outcome == 'folded_preflop':
        return f"Seat {seat}: {name}{role_str} folded before Flop"
    elif outcome == 'folded_flop':
        return f"Seat {seat}: {name}{role_str} folded on the Flop"
    elif outcome == 'folded_turn':
        return f"Seat {seat}: {name}{role_str} folded on the Turn"
    elif outcome == 'folded_river':
        return f"Seat {seat}: {name}{role_str} folded on the River"
    else:
        return f"Seat {seat}: {name}{role_str} {outcome}"
