#!/usr/bin/env python3
"""
Script to check for duplicate callback outputs in Dash app
"""
import re
import sys

def find_duplicate_outputs(file_path):
    """Find duplicate Output IDs in callbacks"""
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find all @app.callback decorators with their line numbers
    callback_pattern = r'@app\.callback\s*\((.*?)\)'
    callbacks = []
    
    for match in re.finditer(callback_pattern, content, re.DOTALL):
        start_pos = match.start()
        line_num = content[:start_pos].count('\n') + 1
        callback_content = match.group(1)
        callbacks.append((line_num, callback_content))
    
    all_outputs = {}
    
    for callback_num, (line_num, callback_content) in enumerate(callbacks, 1):
        # Find all Output() calls - handle both single and multi-line
        output_pattern = r"Output\(['\"]([^'\"]+)['\"][^)]*\)"
        outputs = re.findall(output_pattern, callback_content)
        
        # Also check for allow_duplicate
        for output_match in re.finditer(output_pattern, callback_content):
            output_id = output_match.group(1)
            output_full = output_match.group(0)
            has_allow_duplicate = 'allow_duplicate=True' in output_full
            
            if output_id not in all_outputs:
                all_outputs[output_id] = []
            all_outputs[output_id].append({
                'callback': callback_num,
                'line': line_num,
                'has_allow_duplicate': has_allow_duplicate
            })
    
    # Find duplicates
    duplicates = {k: v for k, v in all_outputs.items() if len(v) > 1}
    
    if duplicates:
        print("❌ DUPLICATE CALLBACK OUTPUTS FOUND:")
        print("=" * 70)
        for output_id, occurrences in duplicates.items():
            print(f"\nOutput ID: '{output_id}'")
            print(f"  Found in {len(occurrences)} callbacks:")
            for occ in occurrences:
                status = "✓" if occ['has_allow_duplicate'] else "✗"
                print(f"    {status} Callback #{occ['callback']} at line {occ['line']} "
                      f"{'(has allow_duplicate=True)' if occ['has_allow_duplicate'] else '(MISSING allow_duplicate=True)'}")
            
            # Check if all have allow_duplicate
            all_have_allow_duplicate = all(occ['has_allow_duplicate'] for occ in occurrences)
            if not all_have_allow_duplicate:
                print(f"  ⚠️  WARNING: Not all occurrences have allow_duplicate=True!")
        print("\n" + "=" * 70)
        return False
    else:
        print("✓ No duplicate callback outputs found!")
        return True

if __name__ == '__main__':
    file_path = 'app.py'
    success = find_duplicate_outputs(file_path)
    sys.exit(0 if success else 1)
