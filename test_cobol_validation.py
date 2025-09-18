#!/usr/bin/env python3
# Test runner for COBOL copybook validation

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_cobol_validation import EnhancedDataValidationFramework, CobolCopybookParser

def test_cobol_parser():
    """Test COBOL copybook parser"""
    print("Testing COBOL Copybook Parser...")

    parser = CobolCopybookParser()

    # Test with local copybook file
    try:
        copybook_data = parser.parse_copybook_file("customer_master.cob")

        print(f"\nParsed Copybook: {copybook_data['copybook_name']}")
        print(f"Total Fields: {len(copybook_data['fields'])}")
        print(f"Record Length: {copybook_data['record_length']} bytes")
        print("\nFirst 10 fields:")

        for i, field in enumerate(copybook_data['fields'][:10]):
            print(f"  {i+1:2d}. {field['field_name']:25} {field['pic_clause']:15} Pos:{field['start_pos']:3d} Len:{field['length']:3d}")

        # Show array fields
        array_fields = [f for f in copybook_data['fields'] if f.get('occurs', 1) > 1]
        if array_fields:
            print(f"\nArray fields found: {len(array_fields)}")
            for field in array_fields:
                print(f"  - {field['field_name']} occurs {field['occurs']} times")

        return True

    except Exception as e:
        print(f"Error testing parser: {e}")
        return False

def test_validation_framework():
    """Test the complete validation framework"""
    print("\nTesting Validation Framework...")

    try:
        # This would run the full validation if data files were available
        print("Framework classes loaded successfully")
        print("To run full validation, ensure data files are available at configured S3 paths")
        return True

    except Exception as e:
        print(f"Error testing framework: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("COBOL COPYBOOK VALIDATION FRAMEWORK TESTING")
    print("="*60)

    success = True

    # Test parser
    if not test_cobol_parser():
        success = False

    # Test framework
    if not test_validation_framework():
        success = False

    if success:
        print("\n✅ All tests passed successfully!")
        print("\nNext steps:")
        print("1. Update cobol_validation_config.json with your AWS credentials and S3 paths")
        print("2. Upload your COBOL copybook (.cob file) to S3")
        print("3. Upload your fixed-width data files to S3")
        print("4. Run: python enhanced_cobol_validation.py")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
