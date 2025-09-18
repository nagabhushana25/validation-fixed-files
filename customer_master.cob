      ******************************************************************
      * CUSTOMER MASTER RECORD COPYBOOK
      * VERSION: 1.0
      * DATE: 2025-09-18
      ******************************************************************
       01  CUSTOMER-MASTER-RECORD.
           05  CUSTOMER-ID                 PIC X(10).
           05  CUSTOMER-INFO.
               10  FIRST-NAME              PIC X(20).
               10  LAST-NAME               PIC X(20).
               10  MIDDLE-INITIAL          PIC X(1).
           05  ADDRESS-INFO.
               10  STREET-ADDRESS          PIC X(40).
               10  CITY                    PIC X(25).
               10  STATE                   PIC X(2).
               10  ZIP-CODE                PIC X(10).
           05  CONTACT-INFO.
               10  PHONE-NUMBERS           OCCURS 3 TIMES.
                   15  PHONE-NUMBER        PIC X(15).
                   15  PHONE-TYPE          PIC X(1).
               10  EMAIL-ADDRESS           PIC X(50).
           05  ACCOUNT-INFO.
               10  ACCOUNT-BALANCE         PIC S9(10)V99 COMP-3.
               10  CREDIT-LIMIT            PIC S9(8)V99.
               10  ACCOUNT-STATUS          PIC X(1).
               10  OPEN-DATE               PIC X(8).
               10  LAST-ACTIVITY-DATE      PIC X(8).
           05  DEMOGRAPHICS.
               10  BIRTH-DATE              PIC X(8).
               10  GENDER                  PIC X(1).
               10  INCOME-RANGE            PIC X(2).
           05  AUDIT-FIELDS.
               10  CREATED-DATE            PIC X(8).
               10  CREATED-TIME            PIC X(6).
               10  UPDATED-DATE            PIC X(8).
               10  UPDATED-TIME            PIC X(6).
               10  RECORD-VERSION          PIC 9(3).
           05  FILLER                      PIC X(25).
