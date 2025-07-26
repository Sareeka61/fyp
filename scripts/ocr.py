devanagari_map = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    'क': 'ka', 'को': 'ko', 'ख': 'kha', 'ग': 'ga', 'च': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'डि': 'ḍi', 'त': 'ta', 'ना': 'na', 'प': 'pa', 'पर': 'par', 'ब': 'ba', 'वा': 'wa', 'भे': 'bhe',
    'म': 'ma', 'मे': 'me', 'य': 'ya', 'लु': 'lu', 'सी': 'si', 'सु': 'su', 'से': 'se', 'ह': 'ha',
}

label_to_index = {char: idx for idx, char in enumerate(devanagari_map.keys())}
index_to_char = {v: k for k, v in label_to_index.items()}
