# Enhanced Data Validation Framework with COBOL Copybook Parser

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import boto3
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import os
import re
from dataclasses import dataclass

@dataclass
class CobolField:
    """Represents a COBOL field definition"""
    level: str
    name: str
    pic_clause: str
    start_pos: int
    length: int
    data_type: str
    decimal_positions: int = 0
    occurs: int = 1
    redefines: Optional[str] = None
    is_pk: bool = False
    parent: Optional[str] = None

class CobolCopybookParser:
    """
    Comprehensive COBOL Copybook Parser for mainframe .cob files
    Handles various COBOL constructs including PIC clauses, OCCURS, REDEFINES, etc.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.fields = []
        self.current_position = 1
        self.level_stack = {}
        
    def parse_copybook_file(self, file_path: str) -> Dict:
        """
        Parse a COBOL copybook file (.cob) from S3 or local filesystem
        
        Args:
            file_path: Path to the COBOL copybook file
            
        Returns:
            Dictionary with parsed field definitions
        """
        try:
            if file_path.startswith('s3://'):
                content = self._read_s3_file(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            return self.parse_copybook_content(content)
            
        except Exception as e:
            self.logger.error(f"Error parsing copybook file {file_path}: {str(e)}")
            raise
    
    def _read_s3_file(self, s3_path: str) -> str:
        """Read file content from S3"""
        import boto3
        
        # Parse S3 path
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj['Body'].read().decode('utf-8')
    
    def parse_copybook_content(self, content: str) -> Dict:
        """
        Parse COBOL copybook content
        
        Args:
            content: Raw copybook content as string
            
        Returns:
            Structured dictionary with field definitions
        """
        self.fields = []
        self.current_position = 1
        self.level_stack = {}
        
        lines = self._preprocess_lines(content)
        
        for line_num, line in enumerate(lines, 1):
            try:
                if self._is_field_definition(line):
                    field = self._parse_field_line(line)
                    if field:
                        self._calculate_field_position(field)
                        self.fields.append(field)
                        self.level_stack[field.level] = field
                        
            except Exception as e:
                self.logger.warning(f"Error parsing line {line_num}: {line.strip()} - {str(e)}")
                continue
        
        # Calculate total record length
        total_length = max([f.start_pos + f.length - 1 for f in self.fields]) if self.fields else 0
        
        copybook_name = self._extract_copybook_name(content)
        
        return {
            "copybook_name": copybook_name,
            "record_length": total_length,
            "fields": [self._field_to_dict(f) for f in self.fields],
            "parsing_summary": {
                "total_fields": len(self.fields),
                "parsed_lines": len(lines),
                "record_length": total_length
            }
        }
    
    def _preprocess_lines(self, content: str) -> List[str]:
        """
        Preprocess copybook lines - handle comments, line numbers, etc.
        """
        lines = []
        
        for line in content.split('\n'):
            # Remove line numbers (columns 1-6) if present
            if len(line) > 6 and line[6] in [' ', '*']:
                line = line[6:]
            
            # Skip comment lines
            if line.strip().startswith('*') or line.strip().startswith('C'):
                continue
                
            # Skip empty lines
            if not line.strip():
                continue
                
            # Handle continuation lines (+ in column 7)
            if len(line) > 0 and line[0] == '+' and lines:
                lines[-1] = lines[-1].rstrip() + ' ' + line[1:].strip()
            else:
                lines.append(line)
        
        return lines
    
    def _is_field_definition(self, line: str) -> bool:
        """Check if line contains a field definition"""
        line = line.strip()
        
        # Must start with level number
        level_match = re.match(r'^\s*(\d{2})\s+', line)
        if not level_match:
            return False
            
        # Must contain field name
        if not re.search(r'^\s*\d{2}\s+[\w-]+', line):
            return False
            
        return True
    
    def _parse_field_line(self, line: str) -> Optional[CobolField]:
        """
        Parse a single COBOL field definition line
        
        Example formats:
        05  CUSTOMER-ID         PIC X(10).
        10  ACCOUNT-BALANCE     PIC 9(10)V99.
        05  PHONE-ARRAY         PIC X(15) OCCURS 5 TIMES.
        05  ALT-NAME            PIC X(30) REDEFINES CUSTOMER-NAME.
        """
        line = line.strip()
        
        # Extract level number
        level_match = re.match(r'^\s*(\d{2})\s+', line)
        if not level_match:
            return None
        
        level = level_match.group(1)
        remaining = line[level_match.end():].strip()
        
        # Extract field name
        name_match = re.match(r'^([\w-]+)', remaining)
        if not name_match:
            return None
        
        field_name = name_match.group(1).replace('-', '_').upper()
        remaining = remaining[name_match.end():].strip()
        
        # Initialize field
        field = CobolField(
            level=level,
            name=field_name,
            pic_clause="",
            start_pos=0,  # Will be calculated later
            length=0,
            data_type="STRING"
        )
        
        # Extract PIC clause
        pic_match = re.search(r'PIC\s+([\w\(\)V\-\.]+)', remaining, re.IGNORECASE)
        if pic_match:
            field.pic_clause = pic_match.group(1)
            field.length, field.data_type, field.decimal_positions = self._parse_pic_clause(field.pic_clause)
        else:
            # Group level - no PIC clause
            field.length = 0
            
        # Extract OCCURS clause
        occurs_match = re.search(r'OCCURS\s+(\d+)', remaining, re.IGNORECASE)
        if occurs_match:
            field.occurs = int(occurs_match.group(1))
            
        # Extract REDEFINES clause
        redefines_match = re.search(r'REDEFINES\s+([\w-]+)', remaining, re.IGNORECASE)
        if redefines_match:
            field.redefines = redefines_match.group(1).replace('-', '_').upper()
        
        return field
    
    def _parse_pic_clause(self, pic_clause: str) -> Tuple[int, str, int]:
        """
        Parse COBOL PIC clause to determine length, data type, and decimal positions
        
        Examples:
        X(10) -> length=10, type=STRING, decimal=0
        9(5) -> length=5, type=INTEGER, decimal=0  
        9(10)V99 -> length=12, type=DECIMAL, decimal=2
        S9(8)V99 -> length=10, type=DECIMAL, decimal=2 (S for signed)
        """
        pic_clause = pic_clause.upper()
        length = 0
        data_type = "STRING"
        decimal_positions = 0
        
        # Handle signed fields
        is_signed = pic_clause.startswith('S')
        if is_signed:
            pic_clause = pic_clause[1:]
        
        # Split on V for decimal fields
        if 'V' in pic_clause:
            integer_part, decimal_part = pic_clause.split('V', 1)
            decimal_positions = self._calculate_pic_length(decimal_part)
            length = self._calculate_pic_length(integer_part) + decimal_positions
            data_type = "DECIMAL"
        else:
            length = self._calculate_pic_length(pic_clause)
            
            if '9' in pic_clause:
                data_type = "INTEGER"
            elif 'X' in pic_clause or 'A' in pic_clause:
                data_type = "STRING"
            else:
                data_type = "STRING"
        
        return length, data_type, decimal_positions
    
    def _calculate_pic_length(self, pic_part: str) -> int:
        """Calculate length from PIC clause part"""
        length = 0
        
        # Handle repeating patterns like X(10) or 9(5)
        repeat_matches = re.findall(r'[XA9]\((\d+)\)', pic_part)
        for match in repeat_matches:
            length += int(match)
        
        # Handle individual characters like XXX or 999
        individual_chars = re.sub(r'[XA9]\(\d+\)', '', pic_part)
        length += len(individual_chars.replace('(', '').replace(')', ''))
        
        return length
    
    def _calculate_field_position(self, field: CobolField):
        """Calculate field position based on previous fields and level hierarchy"""
        
        if field.redefines:
            # REDEFINES fields start at the same position as the redefined field
            redefined_field = next((f for f in self.fields if f.name == field.redefines), None)
            if redefined_field:
                field.start_pos = redefined_field.start_pos
                return
        
        # For group fields (no PIC clause), position is set but length might be 0
        if field.level == '01' or not self.fields:
            field.start_pos = 1
            self.current_position = 1
        else:
            # Find parent level
            parent_level = None
            for level in sorted([f.level for f in self.fields], reverse=True):
                if int(level) < int(field.level):
                    parent_level = level
                    break
            
            if parent_level:
                # Position after the last field at the same or lower level
                same_level_fields = [f for f in self.fields if f.level >= field.level]
                if same_level_fields:
                    last_field = same_level_fields[-1]
                    field.start_pos = last_field.start_pos + last_field.length * last_field.occurs
                else:
                    field.start_pos = self.current_position
            else:
                field.start_pos = self.current_position
        
        # Update current position for next field
        if field.length > 0:
            self.current_position = field.start_pos + field.length * field.occurs
    
    def _extract_copybook_name(self, content: str) -> str:
        """Extract copybook name from content"""
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            # Look for 01 level name
            match = re.search(r'01\s+([\w-]+)', line)
            if match:
                return match.group(1).replace('-', '_').upper()
        
        return "UNKNOWN_COPYBOOK"
    
    def _field_to_dict(self, field: CobolField) -> Dict:
        """Convert CobolField to dictionary format"""
        return {
            "field_name": field.name,
            "level": field.level,
            "start_pos": field.start_pos,
            "length": field.length,
            "data_type": field.data_type,
            "pic_clause": field.pic_clause,
            "decimal_positions": field.decimal_positions,
            "occurs": field.occurs,
            "redefines": field.redefines,
            "is_pk": field.is_pk
        }

class EnhancedDataValidationFramework:
    """
    Enhanced Data Validation Framework with COBOL Copybook support
    """
    
    def __init__(self, config_path: str):
        """Initialize the validation framework with configuration"""
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        self.logger = self._setup_logging()
        self.s3_client = self._setup_s3_client()
        self.copybook_parser = CobolCopybookParser(self.logger)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _create_spark_session(self) -> SparkSession:
        """Create Spark session with optimized configurations"""
        spark_config = self.config['spark']
        
        spark = (SparkSession.builder
                .appName(spark_config['app_name'])
                .master(spark_config['master'])
                .config("spark.executor.memory", spark_config['executor_memory'])
                .config("spark.driver.memory", spark_config['driver_memory'])
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.hadoop.fs.s3a.access.key", self.config['aws']['access_key_id'])
                .config("spark.hadoop.fs.s3a.secret.key", self.config['aws']['secret_access_key'])
                .config("spark.hadoop.fs.s3a.endpoint", f"s3.{self.config['aws']['region']}.amazonaws.com")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.fast.upload", "true")
                .getOrCreate())
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_validation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_s3_client(self):
        """Setup S3 client for file operations"""
        return boto3.client(
            's3',
            aws_access_key_id=self.config['aws']['access_key_id'],
            aws_secret_access_key=self.config['aws']['secret_access_key'],
            region_name=self.config['aws']['region']
        )
    
    def load_cobol_copybook(self, copybook_path: str) -> Dict:
        """
        Load and parse COBOL copybook (.cob file)
        
        Args:
            copybook_path: Path to COBOL copybook file (.cob)
            
        Returns:
            Dictionary containing parsed field definitions
        """
        try:
            self.logger.info(f"Loading COBOL copybook: {copybook_path}")
            copybook_data = self.copybook_parser.parse_copybook_file(copybook_path)
            
            self.logger.info(f"Parsed copybook: {copybook_data['copybook_name']}")
            self.logger.info(f"Total fields: {copybook_data['parsing_summary']['total_fields']}")
            self.logger.info(f"Record length: {copybook_data['record_length']} bytes")
            
            return copybook_data
            
        except Exception as e:
            self.logger.error(f"Error loading COBOL copybook: {str(e)}")
            raise
    
    def create_schema_from_copybook(self, copybook: Dict) -> StructType:
        """
        Create Spark StructType schema from parsed COBOL copybook
        All fields are initially parsed as StringType for validation
        """
        fields = []
        
        for field_def in copybook['fields']:
            field_name = field_def['field_name']
            
            # Handle array fields (OCCURS)
            if field_def.get('occurs', 1) > 1:
                # Create array of string fields
                for i in range(field_def['occurs']):
                    array_field_name = f"{field_name}_{i+1:02d}"
                    fields.append(StructField(array_field_name, StringType(), True))
            else:
                fields.append(StructField(field_name, StringType(), True))
        
        return StructType(fields)
    
    def read_fixed_width_file(self, file_path: str, copybook: Dict) -> DataFrame:
        """
        Read fixed-width file using parsed COBOL copybook definition
        """
        try:
            self.logger.info(f"Reading fixed-width file: {file_path}")
            self.logger.info(f"Using copybook: {copybook['copybook_name']}")
            
            # Read raw text file
            raw_df = self.spark.read.text(file_path)
            
            # Add row numbers for debugging
            raw_df = raw_df.withColumn("row_number", monotonically_increasing_id())
            
            # Extract fields based on copybook positions
            select_expressions = []
            
            for field_def in copybook['fields']:
                field_name = field_def['field_name']
                start_pos = field_def['start_pos']
                length = field_def['length']
                occurs = field_def.get('occurs', 1)
                
                if occurs > 1:
                    # Handle array fields
                    current_pos = start_pos
                    for i in range(occurs):
                        array_field_name = f"{field_name}_{i+1:02d}"
                        select_expr = trim(substring(col("value"), current_pos, length)).alias(array_field_name)
                        select_expressions.append(select_expr)
                        current_pos += length
                else:
                    # Single field
                    if length > 0:  # Skip group fields with no length
                        select_expr = trim(substring(col("value"), start_pos, length)).alias(field_name)
                        select_expressions.append(select_expr)
            
            # Create DataFrame with parsed columns
            df = raw_df.select("row_number", *select_expressions)
            
            # Add metadata columns
            df = df.withColumn("source_file", lit(file_path)) \
                   .withColumn("record_length", length(col("value"))) \
                   .withColumn("expected_length", lit(copybook['record_length'])) \
                   .withColumn("length_match", col("record_length") == col("expected_length")) \
                   .withColumn("load_timestamp", current_timestamp())
            
            record_count = df.count()
            length_mismatches = df.filter(col("length_match") == False).count()
            
            self.logger.info(f"Successfully parsed {record_count:,} records")
            if length_mismatches > 0:
                self.logger.warning(f"Found {length_mismatches:,} records with incorrect length")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading fixed-width file {file_path}: {str(e)}")
            raise
    
    def validate_copybook_structure(self, df: DataFrame, copybook: Dict) -> Dict:
        """
        Validate that the data file matches the copybook structure
        """
        validation_results = {
            "copybook_name": copybook['copybook_name'],
            "expected_length": copybook['record_length'],
            "validation_timestamp": datetime.now().isoformat(),
            "issues": []
        }
        
        # Check record lengths
        length_stats = df.select(
            min("record_length").alias("min_length"),
            max("record_length").alias("max_length"),
            avg("record_length").alias("avg_length")
        ).collect()[0]
        
        validation_results["actual_lengths"] = {
            "min": length_stats.min_length,
            "max": length_stats.max_length,
            "avg": round(length_stats.avg_length, 2)
        }
        
        # Check for length mismatches
        length_mismatches = df.filter(col("length_match") == False).count()
        if length_mismatches > 0:
            validation_results["issues"].append(
                f"{length_mismatches:,} records have incorrect length"
            )
        
        # Check for completely empty records
        empty_records = df.filter(col("record_length") == 0).count()
        if empty_records > 0:
            validation_results["issues"].append(
                f"{empty_records:,} completely empty records found"
            )
        
        # Validate primary key fields
        pk_fields = self.config['validation'].get('primary_keys', [])
        for pk_field in pk_fields:
            if pk_field in df.columns:
                null_pks = df.filter(col(pk_field).isNull() | (col(pk_field) == "")).count()
                if null_pks > 0:
                    validation_results["issues"].append(
                        f"Primary key field {pk_field} has {null_pks:,} null/empty values"
                    )
        
        return validation_results
    
    def compare_dataframes(self, df1: DataFrame, df2: DataFrame, 
                          primary_keys: List[str], ignore_fields: List[str] = None) -> Tuple[DataFrame, DataFrame, Dict]:
        """
        Enhanced DataFrame comparison with COBOL-specific handling
        """
        self.logger.info("Starting enhanced DataFrame comparison...")
        
        if ignore_fields is None:
            ignore_fields = self.config['validation'].get('ignore_fields', [])
        
        # Get common columns for comparison (exclude metadata)
        metadata_cols = {"source_file", "record_length", "expected_length", "length_match", "load_timestamp", "row_number"}
        df1_cols = set(df1.columns) - metadata_cols
        df2_cols = set(df2.columns) - metadata_cols
        common_cols = list((df1_cols & df2_cols) - set(ignore_fields))
        
        self.logger.info(f"Comparing {len(common_cols)} common columns")
        self.logger.info(f"Primary keys: {primary_keys}")
        
        # Identify array fields for special handling
        array_fields = {}
        for col_name in common_cols:
            if '_' in col_name and col_name.split('_')[-1].isdigit():
                base_name = '_'.join(col_name.split('_')[:-1])
                if base_name not in array_fields:
                    array_fields[base_name] = []
                array_fields[base_name].append(col_name)
        
        # Rename columns to avoid conflicts
        df1_renamed = df1.select(*primary_keys, *[col(c).alias(f"{c}_legacy") for c in common_cols if c not in primary_keys])
        df2_renamed = df2.select(*primary_keys, *[col(c).alias(f"{c}_modern") for c in common_cols if c not in primary_keys])
        
        # Full outer join on primary keys
        joined_df = df1_renamed.join(df2_renamed, primary_keys, "full_outer")
        
        # Create detailed comparison logic
        mismatch_conditions = []
        field_comparison_cols = []
        
        for field in common_cols:
            if field not in primary_keys:
                legacy_col = f"{field}_legacy"
                modern_col = f"{field}_modern"
                
                # Get tolerance for numeric fields
                tolerance = self.config['validation'].get('tolerance', {}).get(field, 0)
                
                if tolerance > 0:
                    # Numeric comparison with tolerance
                    try:
                        mismatch_condition = abs(col(legacy_col).cast("double") - col(modern_col).cast("double")) > tolerance
                    except:
                        mismatch_condition = col(legacy_col) != col(modern_col)
                else:
                    # String comparison (handle nulls and empty strings as equivalent)
                    mismatch_condition = (
                        (coalesce(col(legacy_col), lit("")) != coalesce(col(modern_col), lit("")))
                    )
                
                mismatch_conditions.append(
                    when(mismatch_condition, lit(field)).otherwise(lit(None))
                )
                
                # Add detailed comparison result
                field_comparison_cols.append(
                    when(mismatch_condition, 
                         concat(lit(f"{field}:L["), coalesce(col(legacy_col), lit("NULL")), 
                               lit("]!=M["), coalesce(col(modern_col), lit("NULL")), lit("]")))
                    .otherwise(lit(None))
                    .alias(f"{field}_comparison")
                )
        
        # Create mismatch details DataFrame
        mismatch_details = joined_df.select(
            *primary_keys,
            *[coalesce(col(f"{field}_legacy"), lit("")).alias(f"{field}_legacy") for field in common_cols if field not in primary_keys],
            *[coalesce(col(f"{field}_modern"), lit("")).alias(f"{field}_modern") for field in common_cols if field not in primary_keys],
            *field_comparison_cols,
            array_remove(array(*mismatch_conditions), lit(None)).alias("mismatched_fields"),
            size(array_remove(array(*mismatch_conditions), lit(None))).alias("mismatch_count"),
            current_timestamp().alias("validation_timestamp")
        ).filter(col("mismatch_count") > 0)
        
        # Calculate metrics
        total_legacy = df1.count()
        total_modern = df2.count()
        
        legacy_only = joined_df.filter(col(f"{common_cols[0]}_modern").isNull()).count()
        modern_only = joined_df.filter(col(f"{common_cols[0]}_legacy").isNull()).count()
        common_records = joined_df.filter(
            col(f"{common_cols[0]}_legacy").isNotNull() & 
            col(f"{common_cols[0]}_modern").isNotNull()
        ).count()
        
        mismatched_records = mismatch_details.count()
        
        metrics = {
            "total_legacy_records": total_legacy,
            "total_modern_records": total_modern,
            "legacy_only_records": legacy_only,
            "modern_only_records": modern_only,
            "common_records": common_records,
            "mismatched_records": mismatched_records,
            "match_rate_percent": round(((common_records - mismatched_records) / max(common_records, 1)) * 100, 2),
            "comparison_timestamp": datetime.now().isoformat(),
            "compared_fields": common_cols,
            "array_fields_detected": list(array_fields.keys())
        }
        
        # Create enhanced summary DataFrame
        summary_data = [
            ("Total Legacy Records", total_legacy),
            ("Total Modern Records", total_modern),
            ("Records Only in Legacy", legacy_only),
            ("Records Only in Modern", modern_only),
            ("Common Records", common_records),
            ("Mismatched Records", mismatched_records),
            ("Perfect Matches", common_records - mismatched_records),
            ("Match Rate (%)", metrics["match_rate_percent"]),
            ("Fields Compared", len(common_cols)),
            ("Array Fields", len(array_fields))
        ]
        
        summary_df = self.spark.createDataFrame(summary_data, ["Metric", "Value"])
        
        self.logger.info(f"Comparison complete:")
        self.logger.info(f"  - Total records: Legacy={total_legacy:,}, Modern={total_modern:,}")
        self.logger.info(f"  - Common records: {common_records:,}")
        self.logger.info(f"  - Mismatched records: {mismatched_records:,}")
        self.logger.info(f"  - Match rate: {metrics['match_rate_percent']:.2f}%")
        
        return mismatch_details, summary_df, metrics
    
    def run_full_validation(self):
        """
        Execute complete validation workflow with COBOL copybook support
        """
        try:
            self.logger.info("=== Starting COBOL Copybook Data Validation ===")
            
            # Load COBOL copybook
            copybook_path = self.config['s3_paths']['copybook']
            copybook = self.load_cobol_copybook(copybook_path)
            
            # Read data files
            legacy_path = self.config['s3_paths']['legacy_file']
            modern_path = self.config['s3_paths']['modern_file']
            
            self.logger.info("Reading legacy data file...")
            legacy_df = self.read_fixed_width_file(legacy_path, copybook)
            
            self.logger.info("Reading modern data file...")
            modern_df = self.read_fixed_width_file(modern_path, copybook)
            
            # Validate copybook structure
            self.logger.info("Validating copybook structure...")
            legacy_structure = self.validate_copybook_structure(legacy_df, copybook)
            modern_structure = self.validate_copybook_structure(modern_df, copybook)
            
            # Compare dataframes
            primary_keys = self.config['validation']['primary_keys']
            mismatch_df, summary_df, metrics = self.compare_dataframes(
                legacy_df, modern_df, primary_keys
            )
            
            # Write results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = self.config['s3_paths']['output_path']
            
            # Write mismatch details if any
            mismatch_path = None
            if mismatch_df.count() > 0:
                mismatch_path = f"{output_base}/mismatch_details/{timestamp}"
                self.write_results_to_s3(mismatch_df, mismatch_path, "parquet")
                self.logger.info(f"Mismatch details written to: {mismatch_path}")
            
            # Write summary
            summary_path = f"{output_base}/summary/{timestamp}"
            self.write_results_to_s3(summary_df, summary_path, "csv")
            
            # Generate comprehensive report
            report_html = self.generate_enhanced_report(
                metrics, copybook, legacy_structure, modern_structure
            )
            
            with open(f"cobol_validation_report_{timestamp}.html", "w") as f:
                f.write(report_html)
            
            self.logger.info("=== Validation Complete ===")
            
            return {
                "status": "completed",
                "copybook": copybook['copybook_name'],
                "metrics": metrics,
                "structure_validation": {
                    "legacy": legacy_structure,
                    "modern": modern_structure
                },
                "output_paths": {
                    "mismatch_details": mismatch_path,
                    "summary": summary_path,
                    "report": f"cobol_validation_report_{timestamp}.html"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Validation workflow failed: {str(e)}")
            raise
        finally:
            self.spark.stop()
    
    def write_results_to_s3(self, df: DataFrame, s3_path: str, format_type: str = "parquet"):
        """Write results to S3 with enhanced error handling"""
        try:
            self.logger.info(f"Writing {df.count():,} records to {s3_path}")
            
            writer = df.coalesce(1).write.mode("overwrite")
            
            if format_type.lower() == "csv":
                writer.option("header", "true").csv(s3_path)
            elif format_type.lower() == "json":
                writer.json(s3_path)
            else:
                writer.parquet(s3_path)
            
            self.logger.info(f"Successfully wrote data to {s3_path}")
            
        except Exception as e:
            self.logger.error(f"Error writing to S3: {str(e)}")
            raise
    
    def generate_enhanced_report(self, metrics: Dict, copybook: Dict, 
                               legacy_structure: Dict, modern_structure: Dict) -> str:
        """Generate enhanced HTML report with COBOL-specific information"""
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>COBOL Copybook Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                .metric-good {{ color: #27ae60; font-weight: bold; }}
                .metric-warning {{ color: #f39c12; font-weight: bold; }}
                .metric-error {{ color: #e74c3c; font-weight: bold; }}
                .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .info-box {{ background-color: #ecf0f1; padding: 15px; border-left: 5px solid #3498db; margin: 15px 0; }}
                .copybook-info {{ background-color: #f8f9fa; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
                .issue-list {{ background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>COBOL Copybook Data Validation Report</h1>
                
                <div class="summary-box">
                    <h2 style="margin-top: 0; color: white;">Validation Summary</h2>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p><strong>Copybook:</strong> {copybook['copybook_name']}</p>
                            <p><strong>Validation Date:</strong> {metrics['comparison_timestamp']}</p>
                            <p><strong>Record Length:</strong> {copybook['record_length']} bytes</p>
                        </div>
                        <div style="text-align: center;">
                            <h3 style="margin: 0; font-size: 2.5em; color: {'#2ecc71' if metrics['match_rate_percent'] >= 95 else '#f39c12' if metrics['match_rate_percent'] >= 80 else '#e74c3c'}">{metrics['match_rate_percent']:.1f}%</h3>
                            <p style="margin: 5px 0;">Match Rate</p>
                        </div>
                    </div>
                </div>

                <h2>Copybook Information</h2>
                <div class="copybook-info">
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>Copybook Name</td><td>{copybook['copybook_name']}</td></tr>
                        <tr><td>Total Fields</td><td>{copybook['parsing_summary']['total_fields']}</td></tr>
                        <tr><td>Record Length</td><td>{copybook['record_length']} bytes</td></tr>
                        <tr><td>Array Fields</td><td>{len(metrics.get('array_fields_detected', []))}</td></tr>
                    </table>
                </div>

                <h2>Comparison Results</h2>
                <table>
                    <tr><th>Metric</th><th>Legacy File</th><th>Modern File</th><th>Difference</th></tr>
                    <tr>
                        <td>Total Records</td>
                        <td>{metrics['total_legacy_records']:,}</td>
                        <td>{metrics['total_modern_records']:,}</td>
                        <td class="{'metric-good' if metrics['total_legacy_records'] == metrics['total_modern_records'] else 'metric-warning'}">
                            {metrics['total_modern_records'] - metrics['total_legacy_records']:,}
                        </td>
                    </tr>
                    <tr>
                        <td>Common Records</td>
                        <td colspan="2" style="text-align: center;">{metrics['common_records']:,}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Mismatched Records</td>
                        <td colspan="2" style="text-align: center;" class="{'metric-good' if metrics['mismatched_records'] == 0 else 'metric-warning' if metrics['mismatched_records'] < metrics['common_records'] * 0.05 else 'metric-error'}">{metrics['mismatched_records']:,}</td>
                        <td>{(metrics['mismatched_records'] / max(metrics['common_records'], 1) * 100):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Legacy Only</td>
                        <td class="{'metric-good' if metrics['legacy_only_records'] == 0 else 'metric-warning'}">{metrics['legacy_only_records']:,}</td>
                        <td>-</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Modern Only</td>
                        <td>-</td>
                        <td class="{'metric-good' if metrics['modern_only_records'] == 0 else 'metric-warning'}">{metrics['modern_only_records']:,}</td>
                        <td>-</td>
                    </tr>
                </table>

                <h2>Structure Validation</h2>
                
                <h3>Legacy File Structure</h3>
                <div class="info-box">
                    <p><strong>Expected Length:</strong> {legacy_structure['expected_length']} bytes</p>
                    <p><strong>Actual Length Range:</strong> {legacy_structure['actual_lengths']['min']} - {legacy_structure['actual_lengths']['max']} bytes (avg: {legacy_structure['actual_lengths']['avg']})</p>
                </div>
                
                {"<div class='issue-list'><h4>Legacy File Issues:</h4><ul>" + "".join([f"<li class='metric-error'>{issue}</li>" for issue in legacy_structure.get('issues', [])]) + "</ul></div>" if legacy_structure.get('issues') else ""}

                <h3>Modern File Structure</h3>
                <div class="info-box">
                    <p><strong>Expected Length:</strong> {modern_structure['expected_length']} bytes</p>
                    <p><strong>Actual Length Range:</strong> {modern_structure['actual_lengths']['min']} - {modern_structure['actual_lengths']['max']} bytes (avg: {modern_structure['actual_lengths']['avg']})</p>
                </div>
                
                {"<div class='issue-list'><h4>Modern File Issues:</h4><ul>" + "".join([f"<li class='metric-error'>{issue}</li>" for issue in modern_structure.get('issues', [])]) + "</ul></div>" if modern_structure.get('issues') else ""}

                <h2>Field Analysis</h2>
                <p><strong>Fields Compared:</strong> {len(metrics['compared_fields'])}</p>
                {"<p><strong>Array Fields Detected:</strong> " + ", ".join(metrics.get('array_fields_detected', [])) + "</p>" if metrics.get('array_fields_detected') else ""}
                
                <div class="info-box">
                    <p><strong>Recommendation:</strong> 
                    {"Excellent data quality! Files are highly consistent." if metrics['match_rate_percent'] >= 95 
                    else "Good data quality with minor discrepancies. Review mismatch details." if metrics['match_rate_percent'] >= 80
                    else "Significant data discrepancies detected. Detailed investigation required."}
                    </p>
                </div>

                <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
                    <p>Generated by Enhanced Data Validation Framework • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_report

# Usage example for COBOL copybooks
def main():
    """Main execution function for COBOL copybook validation"""
    
    # Create configuration for COBOL copybook
    config = {
        "aws": {
            "access_key_id": "your_access_key",
            "secret_access_key": "your_secret_key",
            "region": "us-east-1"
        },
        "s3_paths": {
            "legacy_file": "s3://your-bucket/legacy/customer_data.txt",
            "modern_file": "s3://your-bucket/modern/customer_data.txt",
            "copybook": "s3://your-bucket/metadata/customer.cob",  # COBOL copybook
            "output_path": "s3://your-bucket/validation_results/"
        },
        "spark": {
            "app_name": "COBOL_DataValidation",
            "master": "local[*]",
            "executor_memory": "4g",
            "driver_memory": "2g"
        },
        "validation": {
            "primary_keys": ["CUSTOMER_ID"],
            "ignore_fields": ["FILLER", "UPDATED_TIMESTAMP"],
            "tolerance": {
                "ACCOUNT_BALANCE": 0.01
            }
        }
    }
    
    # Save configuration
    with open('cobol_validation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize and run validation
    try:
        validator = EnhancedDataValidationFramework("cobol_validation_config.json")
        results = validator.run_full_validation()
        
        print("\n" + "="*60)
        print("COBOL COPYBOOK VALIDATION RESULTS")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Copybook: {results['copybook']}")
        print(f"Match Rate: {results['metrics']['match_rate_percent']:.2f}%")
        print(f"Total Legacy Records: {results['metrics']['total_legacy_records']:,}")
        print(f"Total Modern Records: {results['metrics']['total_modern_records']:,}")
        print(f"Mismatched Records: {results['metrics']['mismatched_records']:,}")
        print(f"Report: {results['output_paths']['report']}")
        print("="*60)
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")

if __name__ == "__main__":
    main()
```