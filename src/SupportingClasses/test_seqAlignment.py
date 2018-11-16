"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import numpy as np
from shutil import rmtree
from unittest import TestCase
from Bio.Align import MultipleSeqAlignment
from SeqAlignment import SeqAlignment


class TestSeqAlignment(TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            SeqAlignment()
        aln_obj1 = SeqAlignment('../Test/1c17A.fa','1c17A')
        self.assertFalse(aln_obj1.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertEqual(aln_obj1.query_id, '>query_1c17A', 'Query ID properly changed per lab protocol.')
        self.assertNotEqual(aln_obj1.query_id, '1c17A', 'Query ID properly changed per lab protocol.')
        self.assertIsNone(aln_obj1.alignment, 'alignment is correctly declared as None.')
        self.assertIsNone(aln_obj1.seq_order, 'seq_order is correctly declared as None.')
        self.assertIsNone(aln_obj1.query_sequence, 'query_sequence is correctly declared as None.')
        self.assertIsNone(aln_obj1.seq_length, 'seq_length is correctly declared as None.')
        self.assertIsNone(aln_obj1.size, 'size is correctly declared as None.')
        self.assertIsNone(aln_obj1.distance_matrix, 'distance_matrix is correctly declared as None.')
        self.assertIsNone(aln_obj1.tree_order, 'tree_order is correctly declared as None.')
        self.assertIsNone(aln_obj1.sequence_assignments, 'sequence_assignments is correctly declared as None.')
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa','1h1vA')
        self.assertFalse(aln_obj2.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertEqual(aln_obj2.query_id, '>query_1h1vA', 'Query ID properly changed per lab protocol.')
        self.assertNotEqual(aln_obj2.query_id, '1h1vA', 'Query ID properly changed per lab protocol.')
        self.assertIsNone(aln_obj2.alignment, 'alignment is correctly declared as None.')
        self.assertIsNone(aln_obj2.seq_order, 'seq_order is correctly declared as None.')
        self.assertIsNone(aln_obj2.query_sequence, 'query_sequence is correctly declared as None.')
        self.assertIsNone(aln_obj2.seq_length, 'seq_length is correctly declared as None.')
        self.assertIsNone(aln_obj2.size, 'size is correctly declared as None.')
        self.assertIsNone(aln_obj2.distance_matrix, 'distance_matrix is correctly declared as None.')
        self.assertIsNone(aln_obj2.tree_order, 'tree_order is correctly declared as None.')
        self.assertIsNone(aln_obj2.sequence_assignments, 'sequence_assignments is correctly declared as None.')

    def test_import_alignment(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        for save in [None, '../Test/1c17A_aln.pkl']:
            aln_obj1.import_alignment(save_file=save)
            self.assertFalse(aln_obj1.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(aln_obj1.query_id, '>query_1c17A', 'Query ID properly changed per lab protocol.')
            self.assertNotEqual(aln_obj1.query_id, '1c17A', 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(aln_obj1.alignment, MultipleSeqAlignment, 'alignment is correctly declared as None.')
            self.assertEqual(aln_obj1.seq_order, ['Q3J6M6-1', 'A4BNZ9-1', 'Q0EZP2-1', 'Q31DL5-1', 'D3LWU0-1',
                                                  'A0P4Z7-1', 'B8AR76-1', 'G2J8E3-1', 'A4STP8-1', 'C5WC20-1',
                                                  'Q8D3J8-1', 'Q89B44-1', 'B8D8G8-1', 'Q494C8-1', 'Q7VQW1-1',
                                                  'Q1LTU9-1', 'A0KQY3-1', 'B2VCA9-1', 'query_1c17A', 'G9EBA7-1',
                                                  'H0J1A3-1', 'A4B9C5-1', 'H3NVB3-1', 'B5JT14-1', 'G9ZC44-1',
                                                  'I8TEE1-1', 'Q6FFK5-1', 'Q8DDH3-1', 'S3EH80-1', 'Q0I5W8-1',
                                                  'A3N2U9-1', 'Q9CKW5-1', 'Q5QZI1-1', 'R7UBD3-1', 'W6LXX8-1',
                                                  'H5SE71-1', 'A1SBU5-1', 'Q48BG0-1', 'S2KJX1-1', 'F7RWD9-1',
                                                  'R9PGI3-1', 'X7E8G8-1', 'B0TQF9-1', 'R8AS80-1', 'K1Z6P6-1',
                                                  'K2CTJ5-1', 'K2D5F5-1', 'A8PQE9-1', 'K2AG65-1'],
                             'seq_order imported correctly.')
            self.assertEqual(aln_obj1.query_sequence, 'MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFLEGAARQPDLIPLLRTQFFIVMGLVDAIP'
                                                      'MIAVGLGLYVMFAVA', 'Query sequence correctly identified.')
            self.assertEqual(aln_obj1.seq_length, 79, 'seq_length is correctly determined.')
            self.assertEqual(aln_obj1.size, 49, 'size is correctly determined.')
            self.assertIsNone(aln_obj1.distance_matrix, 'distance_matrix is correctly declared as None.')
            self.assertIsNone(aln_obj1.tree_order, 'tree_order is correctly declared as None.')
            self.assertIsNone(aln_obj1.sequence_assignments, 'sequence_assignments is correctly declared as None.')
            if save is None:
                self.assertFalse(os.path.isfile(os.path.abspath('../Test/1c17A_aln.pkl')), 'No save performed.')
            else:
                self.assertTrue(os.path.isfile(os.path.abspath('../Test/1c17A_aln.pkl')), 'Save file made correctly.')
                os.remove(os.path.abspath('../Test/1c17A_aln.pkl'))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        for save in [None, '../Test/1h1vA_aln.pkl']:
            aln_obj2.import_alignment(save_file=save)
            self.assertFalse(aln_obj2.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(aln_obj2.query_id, '>query_1h1vA', 'Query ID properly changed per lab protocol.')
            self.assertNotEqual(aln_obj2.query_id, '1h1vA', 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(aln_obj2.alignment, MultipleSeqAlignment,'alignment is correctly initialized.')
            self.assertEqual(aln_obj2.seq_order, ['W4Z7Q7-1', 'K3WHZ1-1', 'A0A067CNU4-1', 'W4FYX5-1', 'D0LWX4-1',
                                                  'R7UQM5-1', 'F0ZSV3-1', 'L7FJ78-1', 'D2UYT9-1', 'G0U265-1',
                                                  'B6KS69-1', 'D2VZ68-1', 'J9ITS7-1', 'D2VH92-1', 'W4YXM8-1',
                                                  'A0E1S8-1', 'L7FL22-1', 'F4PSD7-1', 'D2VNZ3-1', 'F4PVY9-1',
                                                  'D2VUN6-1', 'F6VIH9-1', 'Q9D9J3-1', 'G3TNP4-1', 'G3UJI6-1',
                                                  'H0X7C3-1', 'H0XHD2-1', 'H0XQE9-1', 'H0XJZ2-1', 'H0X8C7-1',
                                                  'H0XXS0-1', 'F6RBR7-1', 'U3KP83-1', 'G1SHW3-1', 'L8Y648-1',
                                                  'G3SNC6-1', 'Q8TDY3-1', 'Q9D9L5-1', 'L5KGB5-1', 'L8YCU9-1',
                                                  'M3WIX4-1', 'F1RJB0-1', 'Q2TA43-1', 'M3Z8W5-1', 'F7DVW3-1',
                                                  'S9WE56-1', 'E2QWP5-1', 'F6WN44-1', 'F7FG86-1', 'Q8TDG2-1',
                                                  'L8GXM8-1', 'D2W078-1', 'F4Q8W5-1', 'W4Z0L8-1', 'W4Z7J1-1',
                                                  'D2VKG9-1', 'I1C368-1', 'H2ZCW0-1', 'L8HDP5-1', 'X6N716-1',
                                                  'D2V8A6-1', 'D3AYM2-1', 'J9ER42-1', 'D2UYR7-1', 'B7P258-1',
                                                  'B0DKC7-1', 'F7EG46-1', 'D3B0W3-1', 'D2VZ15-1', 'R7UXI8-1',
                                                  'E3N7U2-1', 'D3BBZ6-1', 'D3BG26-1', 'X6M6K0-1', 'M2W7W2-1',
                                                  'B5DKX0-1', 'Q54L54-1', 'P51775-1', 'V6LIA4-1', 'F6Y1F4-1',
                                                  'D2VUK3-1', 'Q17C85-1', 'I1G8G1-1', 'I1GHS9-1', 'F4PGJ9-1',
                                                  'A8PB10-1', 'R7URM9-1', 'X6MVH5-1', 'F0ZR57-1', 'Q54PQ2-1',
                                                  'D3B8T6-1', 'F4PYF0-1', 'Q9XZB9-1', 'D2V0P1-1', 'F4Q8W4-1',
                                                  'C3ZM16-1', 'F2E730-1', 'Q9BIG6-1', 'Q95UN9-1', 'P12715-1',
                                                  'P53468-1', 'D3B1Y2-1', 'X6LZP1-1', 'I1GHT0-1', 'Q55CU2-1',
                                                  'Q54HE9-1', 'D2VU98-1', 'W4XW06-1', 'A8YGN0-1', 'E1CFD8-1',
                                                  'T1J8A7-1', 'F5AMM3-1', 'Q76L02-1', 'B6RB19-1', 'B5DJ22-1',
                                                  'D3AW82-1', 'Q76L01-1', 'F2UCJ6-1', 'A8Y226-1', 'T1J9I3-1',
                                                  'C3ZLQ0-1', 'C3ZMR7-2', 'D3BSY8-1', 'F4PJ19-1', 'E3MAX5-1',
                                                  'K1RBG6-1', 'F4PJ20-1', 'W4Z0B2-1', 'Q5QEI8-1', 'K1RVE3-2',
                                                  'R1F3T3-1', 'F2UCJ7-1', 'G0PAI7-1', 'C3XYD5-1', 'H2KRT9-1',
                                                  'G3RV12-1', 'Q8SWN8-1', 'R0M8Z2-1', 'T0MEN7-1', 'L2GXI8-1',
                                                  'Q76L00-1', 'V2Y4Y3-1', 'S7W9W5-1', 'F6ZHA7-1', 'B7XHF2-1',
                                                  'L2GLN4-1', 'L8GEM2-1', 'L8GYN8-1', 'K1RVE3-1', 'R7U3X6-1',
                                                  'K1R557-1', 'J9A0F1-1', 'M3Y8P7-1', 'Q54HF0-1', 'F7GD92-1',
                                                  'F0Z7P3-1', 'F2UCK1-1', 'J0D8Q2-1', 'K1RHJ4-1', 'K1QXE8-1',
                                                  'G3SAF6-1', 'K1QMP4-1', 'W2T202-1', 'C3ZMR7-1', 'C3ZTX5-1',
                                                  'Q54HF1-1', 'X6N7F5-1', 'X6NZM5-1', 'Q2U7A3-1', 'C1GG58-1',
                                                  'Q9P4D1-1', 'N1RMQ6-1', 'H1UXN5-1', 'Q4WDH2-1', 'A0A017S1P3-1',
                                                  'P53455-1', 'U7PLB6-1', 'N1RL78-1', 'Q554S6-1', 'V4BV77-1',
                                                  'S9W2X5-1', 'K1RA57-1', 'F0Z7P2-1', 'I3EP48-1', 'P07828-1',
                                                  'K1RU04-1', 'G4TBF6-1', 'Q9Y896-1', 'B4G2W4-1', 'U4U6N7-1',
                                                  'C3ZMR5-1', 'C3ZMR4-1', 'A8E073-1', 'S9XQ78-1', 'A8PB07-1',
                                                  'D6WJW7-1', 'V2WIQ9-1', 'V2Y4Z9-1', 'B0DKF0-1', 'F6R0S4-1',
                                                  'G3TAI8-1', 'G1S9C9-1', 'G7MTS6-1', 'L8WKN6-1', 'W4YKQ1-1',
                                                  'M2PKY1-1', 'F6SC57-1', 'Q93132-1', 'A7SJK6-1', 'W5JXB3-1',
                                                  'G3VC64-1', 'Q58DT9-1', 'W5PKC6-1', 'F4PJ18-1', 'R9P1L9-1',
                                                  'A0A067QDX1-1', 'E9H3B3-1', 'W2SXZ4-1', 'S9XWJ6-1', 'A8WS37-1',
                                                  'Q9UVF3-1', 'S9XSM0-1', 'P53483-1', 'P11426-1', 'Q8WPD5-1',
                                                  'K1QHY0-1', 'B4G5W8-1', 'L0R5C4-1', 'G3RQT5-1', 'Q6S8J3-1',
                                                  'D3B1V2-1', 'G3QF51-1', 'S9WUG6-1', 'F2DZG9-1', 'A5A6N1-1',
                                                  'H2KPA2-1', 'Q03341-1', 'W6U4L6-1', 'G7Y3S9-1', 'L8WKN1-1',
                                                  'Q9UVX4-1', 'G7E609-1', 'H2Z8N7-1', 'G7YDY9-1', 'W5J4N5-1',
                                                  'H3F5B4-1', 'W4Y157-1', 'P83967-1', 'G7YCA9-1', 'F1L3U5-1',
                                                  'H9J7T4-1', 'F7GR39-1', 'P60709-1', 'L9L479-1', 'P63261-1',
                                                  'S5M4F4-1', 'H2KR85-1', 'K7A690-1', 'P62736-1', 'F6PLA9-1',
                                                  'P63267-1', 'S7MYN0-1', 'query_1h1vA', 'Q5T8M7-1', 'P68133-1',
                                                  'H2R815-1', 'P68032-1', 'G6DIM9-1', 'H9IXU9-1', 'F7FMI7-1',
                                                  'Q6AY16-1', 'Q8CG27-1', 'M3X2X7-1', 'F7FGP2-1', 'Q8TC94-1',
                                                  'S7MH42-1', 'H0XP60-1', 'E2RML1-1', 'Q2T9W4-1', 'L5K8R3-1',
                                                  'F1SA46-1', 'T0MIL1-1', 'L5KV92-1', 'E1B7X2-1', 'G3UNF4-1',
                                                  'G1U4K9-1', 'H0Y1X4-1', 'I3LJA4-1', 'F6XB70-1', 'L5LE08-1',
                                                  'Q9QY84-1', 'Q4R6Q3-1', 'Q9Y615-1', 'L9KP75-1', 'S9Y9E0-1',
                                                  'E2R497-1', 'Q32KZ2-1', 'F1SP29-1', 'E2R4A0-1', 'F6XB87-1',
                                                  'L5LG13-1', 'G1TA65-1', 'Q9QY83-1', 'F6RBJ9-1', 'G3TN84-1',
                                                  'Q32L91-1', 'Q95JK8-1', 'Q9Y614-1', 'D2V3Y5-1', 'Q3M0X2-1',
                                                  'G0QIW4-1', 'I7LW62-1', 'A2FH22-1', 'G0UYV8-1', 'Q387T4-1',
                                                  'G0U4W2-1', 'K4DXF6-1', 'J9JB44-1', 'K8EMB1-1', 'A0A061RHG9-1',
                                                  'D8MBI2-1', 'L1JCH3-1', 'S0AWX7-1', 'S0B0L7-1', 'A8ISF0-1',
                                                  'D8UFJ7-1', 'I0Z7H8-1', 'A4SBH2-1', 'Q00VM6-1', 'F0WFS9-1',
                                                  'H3G9P8-1', 'K3W8L9-1', 'A0A024T9P2-1', 'A0A067C9P1-1', 'D7FV64-1',
                                                  'F0XXQ8-1', 'V7CT22-1', 'W1PF76-1', 'M1BSY7-1', 'B8BA93-1',
                                                  'Q6Z256-1', 'V4K5E9-1', 'V4U305-1', 'M4E8Y7-1', 'Q9LSD6-1',
                                                  'A9RMT8-1', 'D8S0E5-1', 'D8SB34-1', 'E4YJF0-1', 'E5S2L6-1',
                                                  'D2V9N2-1', 'D2VJG9-1', 'R1EGT7-1', 'A0A058ZB61-1', 'P53487-1',
                                                  'F4PQD5-1', 'O96621-1', 'B4G8J6-1', 'U5H6J3-1', 'M7NWC2-1',
                                                  'B6JZD3-1', 'Q9UUJ1-1', 'R4XCF6-1', 'G1X8M6-1', 'S8C0I0-1',
                                                  'D5GHR3-1', 'M2LXB0-1', 'K9GB28-1', 'Q5BFK7-1', 'A1CPC5-1',
                                                  'B6H4Z8-1', 'K3VCA8-1', 'L7JRP4-1', 'F7W7D5-1', 'S3CPY6-1',
                                                  'A0A060T738-1', 'Q6C1K7-1', 'Q759G0-1', 'I2H5H9-1', 'P32381-1',
                                                  'F2QWC6-1', 'A3LYA7-1', 'G3AXQ5-1', 'I1BX49-1', 'I4Y6F7-1',
                                                  'G4TJB5-1', 'G7E9P6-1', 'M7WPT6-1', 'A8PW20-1', 'M5EB81-1',
                                                  'A0A066VYB8-1', 'V5EVF4-1', 'J5R510-1', 'E3JXI2-1', 'F4RKV7-1',
                                                  'L8WPT5-1', 'M5BPW3-1', 'M5GC69-1', 'A0A067MV70-1', 'V2XB33-1',
                                                  'W4K143-1', 'D8Q243-1', 'A0A067TPG6-1', 'R7SGD9-1', 'A0A060SBN4-1',
                                                  'B0CTC7-1', 'A8N9D5-1', 'G4VBW8-1', 'T1KX76-1', 'H3EWD5-1',
                                                  'U1P378-1', 'P53489-1', 'U6NJ96-1', 'F2UIC6-1', 'A9V6V2-1',
                                                  'F4PDS1-1', 'E9C5M3-1', 'H2Z5X4-1', 'V4A503-1', 'T1FP77-1',
                                                  'B3RJY8-1', 'R7TR89-1', 'T2MFE1-1', 'F6QD48-1', 'I1FLA9-1',
                                                  'F7DAH7-1', 'A7S7W1-1', 'G3SJX1-1', 'S7NN93-1', 'P61160-1',
                                                  'W5QFG6-1', 'C4A0H4-1', 'K1QFS4-1', 'A0A023NL46-1', 'J9K2X7-1',
                                                  'E9G618-1', 'B4R5U8-1', 'W8AY62-1', 'P45888-1', 'H9IS56-1',
                                                  'R4WT27-1', 'A0A023BA72-1', 'R7Q985-1', 'Q3M0X7-1', 'Q3M0X9-1',
                                                  'G0R3U5-1', 'Q24C02-1', 'A0A058Z6W9-1', 'W1Q741-1', 'J9F9Y8-1',
                                                  'F2QNY4-1', 'O94630-1', 'B5Y5B2-1', 'B8C9E2-1', 'K0RM55-1',
                                                  'B6A9I8-1', 'Q5CRL6-1', 'D2VQU2-1', 'W7K197-1', 'D8LPW8-1',
                                                  'W7U6E6-1', 'A0A024U2D6-1', 'F0WRQ9-1', 'D0MTM6-1', 'H3G8F5-1',
                                                  'C4Y9V7-1', 'A5DQ76-1', 'G3B2V8-1', 'G8Y7R5-1', 'C5M8G8-1',
                                                  'M3IJA9-1', 'G3APJ3-1', 'B5RTY7-1', 'A3LVF8-1', 'A5E2K2-1',
                                                  'H8X0P7-1', 'A0A061AJ95-1', 'Q54I79-1', 'D3B962-1', 'F4QEP2-1',
                                                  'E9C4W5-1', 'B6JVI6-1', 'F2UJ75-1', 'A9V5L4-1', 'W6MH92-1',
                                                  'B9PQT4-1', 'U6KLL3-1', 'Q6C6Y2-1', 'A0A060T2S9-1', 'A8Q3A0-1',
                                                  'M5EAS4-1', 'U1HFI3-1', 'K9GI51-1', 'V5G412-1', 'H6C2R9-1','W2RYH4-1',
                                                  'Q2GY63-1', 'F9WW78-1', 'M2N0J1-1', 'B2VYQ0-1', 'E5ADV7-1',
                                                  'K2RBZ3-1', 'R7Z3L8-1', 'D4ATY3-1', 'D5GPJ5-1', 'G1XP43-1',
                                                  'U7Q578-1', 'L8G2I3-1', 'N1JC81-1', 'W3X8Q2-1', 'L7JGL5-1',
                                                  'B2B7W3-1', 'P38673-1', 'C9S6W6-1', 'N1RLV5-1', 'E9E4X9-1',
                                                  'A0A063BT17-1', 'F4NW25-1', 'U5HBM6-1', 'G7E6C4-1', 'P42023-1',
                                                  'A0A066VBJ4-1', 'U9UKT3-1', 'S2JIV6-1', 'A0A061HA90-1', 'V5EB15-1',
                                                  'Q4PE63-1', 'R9P4N6-1', 'R9AE54-1', 'M5G835-1', 'Q5KPW4-1',
                                                  'G4TCU4-1', 'A8N1N8-1', 'J4HWB6-1', 'L8WVM7-1', 'E4X3K5-1',
                                                  'L8HDV5-1', 'I1GBC4-1', 'H3FGG2-1', 'U1NUA4-1', 'W2T4S4-1',
                                                  'E5SG12-1', 'J9K0J3-1', 'W6V0K4-1', 'G4LW08-1', 'H2KNF2-1',
                                                  'F6TVZ8-1', 'H2Y6D4-1', 'H2Y6D5-1', 'T1JS59-1', 'C3ZIN7-1',
                                                  'T1FN22-1', 'E9HMX4-1', 'P45889-1', 'W5JLB7-1', 'E0VYK5-1',
                                                  'B7QDE8-1', 'G6D508-1', 'Q1HQC8-1', 'C1BUD7-1', 'E2A6V0-1',
                                                  'T1ISG9-1', 'W4XDB1-1', 'G3N132-1', 'F6ZWP5-1', 'P42025-1',
                                                  'L8Y915-1', 'R4GMT0-1', 'L5MBG6-1', 'L8Y993-1', 'P61163-1',
                                                  'V4AQ12-1', 'A7RID7-1', 'K1Q811-1', 'R7UT93-1', 'J7S8W4-1',
                                                  'G8BXK9-1', 'C5DZM6-1', 'S6EDH7-1', 'G0W6S0-1', 'G0VA67-1',
                                                  'H2AWA6-1', 'I2GXU8-1', 'J8PMY8-1', 'P38696-1', 'J8QFR4-1',
                                                  'C5DCU7-1', 'Q6CM53-1', 'G8JUI0-1', 'R9XKT5-1', 'C1EDX6-1',
                                                  'D8R5L0-1', 'Q84M92-1', 'W1NX64-1', 'A0A067J8R0-1', 'M1BYV2-1',
                                                  'A0A061IV08-1', 'Q4D1Q8-1', 'M2XCM8-1', 'D2VAP1-1', 'F0ZKF6-1',
                                                  'S0B5B4-1', 'C4M3T9-1', 'L0AVA8-1', 'J4C338-1', 'Q4N681-1',
                                                  'A7AQ08-1', 'A0A061CYT9-1', 'C1L3T0-1', 'D2VSL4-1', 'A2DBN6-1',
                                                  'G7YTM5-2', 'Q55DS6-1', 'X1WIR2-1', 'V3Z137-1', 'V4B6R7-1',
                                                  'D2VWT0-1', 'D2VC45-1', 'E4X651-1', 'C3ZLM1-1', 'J9J079-1',
                                                  'A0A067MHL9-1', 'A0A067MRJ5-1', 'A2ECQ8-1', 'D2W474-1', 'S9XKB1-1',
                                                  'L9LB89-1', 'L5KNB5-1', 'J9JH95-1', 'F6YZV6-1', 'G3W1M0-1',
                                                  'Q6BG22-1', 'K7E6E7-1', 'G3VN48-1', 'Q8BXF8-1', 'H0X7T9-1',
                                                  'G1P2S5-1', 'G3TKM5-1', 'Q9BYD9-1', 'M3Z068-1', 'Q2TBH3-1',
                                                  'G1SJB1-1', 'I3LTR7-1', 'L5KP75-1', 'F6VG50-1', 'X6NLU7-1',
                                                  'D2VIR3-1', 'D2V3F0-1', 'R7T5B0-1', 'D2V093-1', 'D2VE55-1',
                                                  'D2VEC2-1', 'A0A061J480-1', 'Q4CV94-1', 'A2E502-1', 'M1V7D1-1',
                                                  'D2VGH4-1', 'A2DKQ5-1', 'E4XMR2-1', 'Q3M0W9-1', 'G7YTM5-1',
                                                  'G4VL77-1', 'Q5DGJ8-1', 'D2V393-1', 'D2VGH8-1', 'Q28WQ3-1',
                                                  'B4MRY2-1', 'B4JWP6-1', 'B4KSP4-1', 'B3MDI2-1', 'P45891-1',
                                                  'D2VJ60-1', 'Q3M0X5-1', 'D2V0N9-1', 'G0R1H2-1', 'I7LWH9-1',
                                                  'P20360-1', 'D8TK19-1', 'O24426-1', 'A0A024VSB6-1', 'W6UF88-1',
                                                  'A0A023BCQ8-1', 'K7UYX2-1', 'W1NVU4-1', 'V4TZQ9-1', 'B9RR79-1',
                                                  'D7TCC4-1', 'U5GD97-1', 'A0A061GL34-1', 'Q6BG21-1', 'A0E660-1',
                                                  'D9N2U7-1', 'D9N2U8-1', 'A0A022R5U7-1', 'A0A059B3W3-1', 'D7LI17-1',
                                                  'P93738-1', 'R7QEX9-1', 'M2XRF6-1', 'Q3KSZ3-1', 'Q948X5-1',
                                                  'P53499-1', 'B5MEK7-1', 'B5MEK8-1', 'D8LXQ7-1', 'A0A022RAT7-1',
                                                  'A0A022R911-1', 'I2FHA6-1', 'I2FHE0-1', 'W7X4V0-1', 'I7M741-1',
                                                  'A0A023B3J4-1', 'D8M7Z9-1', 'S9VIR3-1', 'W6KV94-1', 'P45520-1',
                                                  'P53477-1', 'F2DIT4-1', 'M2XZ77-1', 'D8LXR4-1', 'D8M0Z4-1',
                                                  'D7LI18-1', 'R0HGB1-1', 'A0A059F032-1', 'D2V8I2-1', 'I1TEC2-1',
                                                  'G0R1C5-1', 'P10992-1', 'Q4YU79-1', 'V4LXM5-1', 'O65204-1',
                                                  'Q4W4T1-1', 'Q76IH7-1', 'Q8RYC2-1', 'R0HFC0-1', 'P53500-1',
                                                  'M2W5E2-1', 'X6N6X6-1', 'M0S644-1', 'Q6F5I1-1', 'B8LQ86-1',
                                                  'A4S825-1', 'Q9SWF3-1', 'P23344-1', 'J3M7K7-1', 'W4ZTI7-1',
                                                  'A2WXB0-1', 'O65314-1', 'M4E0Q9-1', 'Q96292-1', 'Q2QLI5-1',
                                                  'W4ZPJ3-1', 'K4AQE7-1', 'A0A059ATR0-1', 'M4CEE0-1', 'M0S1H8-1',
                                                  'M8CE78-1', 'M4CLU1-1', 'M0RLH0-1', 'K4AL27-1', 'P53496-1',
                                                  'B4F989-1', 'M4CXY4-1', 'I1QXY9-1', 'M0S856-1', 'M0TIQ0-1',
                                                  'D2VEW0-1', 'D2VE41-1', 'D2VUS6-1', 'P27131-1', 'D2VS98-1',
                                                  'Q9NJV4-1', 'Q2HX31-1', 'B7G878-1', 'K0SCC3-1', 'A7AX51-1',
                                                  'I7IRD5-1', 'Q8I4X0-1', 'W7JQC5-1', 'C5K5X4-1', 'P26183-1',
                                                  'P22132-1', 'P22131-1', 'C5KIZ2-1', 'C5LNQ3-1'],
                             'seq_order imported correctly.')
            self.assertEqual(aln_obj2.query_sequence, '-------------TTALVCDNGSG----------------LVK-AGFA-------G--D-----'
                                                      '-------------------D-APR-A-----------V----------------------F---'
                                                      '-----------------------------------P-----SIV-GR-P------------R--'
                                                      '---H------------------M-VGM-------------------------------------'
                                                      '-------GQK-----D----------------S--Y-V--------------------------'
                                                      '----------GDEA--QS------------------------KR----G-IL-----------T'
                                                      'LKY-PIEH-G-I-IT-N--WDDMEKIWHHTFY-N--------------ELR-V-----A-----'
                                                      '---P-----------------------------EE-H-PTLLTEAPL---NP-----K----AN'
                                                      'REK----------MTQ-IMFETFNVP--------AMYVA--IQAVLSLYAS------------G'
                                                      '-R-TTGIVLD-SGDGVTHN-VPI-YEGYA-LPHAIMR-LD-LAG--------RD-LTDYL--MK'
                                                      '---IL-TER---------------------G------------------------------Y--'
                                                      '-S------------------F----------------V-TTAE---------------------'
                                                      '----------------------------------REIVRDIK----EK---LCYV---------'
                                                      '----A-LDFEN-----E-MA-------------------TAASS----------SSL-E-KS--'
                                                      '------------YE----------------LPDGQ-----------------------------'
                                                      '---------------VI-----------T---------------------IG-NE-R-FRCPET'
                                                      'LF---------------Q---P-SFI----G-ME---------S---AGI--------------'
                                                      '-HE--------------------TTYNSIM----------K--C-D-ID--I-R----------'
                                                      '----KDLYAN--NVMS-GGTTMYPG-------------------------IA-DRMQKE-I---'
                                                      'TALAP------------------------S-T-------------M--K--I------------'
                                                      '---------------K---------------------------------------------II-'
                                                      '--AP---PE---------RK--YSV-WIGGSIL-ASL---------------------------'
                                                      '-ST-------FQ----Q------------------------------M-WITKQEYD-----EA'
                                                      'G-P-----S--I---VH----R-KCF--------',
                             'Query sequence correctly identified.')
            self.assertEqual(aln_obj2.seq_length, 1506, 'seq_length is correctly determined.')
            self.assertEqual(aln_obj2.size, 785, 'size is correctly determined.')
            self.assertIsNone(aln_obj2.distance_matrix, 'distance_matrix is correctly declared as None.')
            self.assertIsNone(aln_obj2.tree_order, 'tree_order is correctly declared as None.')
            self.assertIsNone(aln_obj2.sequence_assignments, 'sequence_assignments is correctly declared as None.')
            if save is None:
                self.assertFalse(os.path.isfile(os.path.abspath('../Test/1h1vA_aln.pkl')), 'No save performed.')
            else:
                self.assertTrue(os.path.isfile(os.path.abspath('../Test/1h1vA_aln.pkl')), 'Save file made correctly.')
                os.remove(os.path.abspath('../Test/1h1vA_aln.pkl'))

    def test_write_out_alignment(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.write_out_alignment(os.path.abspath('../Test/test_1c17A.fa'))
        aln_obj1.import_alignment()
        aln_obj1.write_out_alignment(os.path.abspath('../Test/test_1c17A.fa'))
        self.assertTrue(os.path.isfile(os.path.abspath('../Test/test_1c17A.fa')), 'Alignment written to correct file.')
        aln_obj1_prime = SeqAlignment(os.path.abspath('../Test/test_1c17A.fa'), '1c17A')
        aln_obj1_prime.import_alignment()
        # self.assertEqual(aln_obj1.alignment, aln_obj1_prime.alignment)
        self.assertEqual(aln_obj1.seq_order, aln_obj1_prime.seq_order)
        self.assertEqual(aln_obj1.query_sequence, aln_obj1_prime.query_sequence)
        self.assertEqual(aln_obj1.seq_length, aln_obj1_prime.seq_length)
        self.assertEqual(aln_obj1.size, aln_obj1_prime.size)
        os.remove(os.path.abspath('../Test/test_1c17A.fa'))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2.write_out_alignment(os.path.abspath('../Test/test_1h1vA.fa'))
        aln_obj2.import_alignment()
        aln_obj2.write_out_alignment(os.path.abspath('../Test/test_1h1vA.fa'))
        self.assertTrue(os.path.isfile(os.path.abspath('../Test/test_1h1vA.fa')), 'Alignment written to correct file.')
        aln_obj2_prime = SeqAlignment(os.path.abspath('../Test/test_1h1vA.fa'), '1h1vA')
        aln_obj2_prime.import_alignment()
        # self.assertEqual(aln_obj2.alignment, aln_obj2_prime.alignment)
        self.assertEqual(aln_obj2.seq_order, aln_obj2_prime.seq_order)
        self.assertEqual(aln_obj2.query_sequence, aln_obj2_prime.query_sequence)
        self.assertEqual(aln_obj2.seq_length, aln_obj2_prime.seq_length)
        self.assertEqual(aln_obj2.size, aln_obj2_prime.size)
        os.remove(os.path.abspath('../Test/test_1h1vA.fa'))

    def test__subset_columns(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1._subset_columns([range(5) + range(745, 749)])
        aln_obj1.import_alignment()
        # One position
        aln_obj1_alpha = aln_obj1._subset_columns([0])
        self.assertEqual(len(aln_obj1_alpha), aln_obj1.size)
        for rec in aln_obj1_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[0])
        aln_obj1_beta = aln_obj1._subset_columns([aln_obj1.seq_length - 1])
        self.assertEqual(len(aln_obj1_beta), aln_obj1.size)
        for rec in aln_obj1_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length - 1])
        aln_obj1_gamma = aln_obj1._subset_columns([aln_obj1.seq_length // 2])
        self.assertEqual(len(aln_obj1_gamma), aln_obj1.size)
        for rec in aln_obj1_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2])
        # Single Range
        aln_obj1_delta = aln_obj1._subset_columns(range(5))
        self.assertEqual(len(aln_obj1_delta), aln_obj1.size)
        for rec in aln_obj1_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[:5])
        aln_obj1_epsilon = aln_obj1._subset_columns(range(aln_obj1.seq_length - 5, aln_obj1.seq_length))
        self.assertEqual(len(aln_obj1_epsilon), aln_obj1.size)
        for rec in aln_obj1_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[-5:])
        aln_obj1_zeta = aln_obj1._subset_columns(range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj1_zeta), aln_obj1.size)
        for rec in aln_obj1_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2:
                                                                       aln_obj1.seq_length // 2 + 5])
        # Mixed Range and Single Position
        aln_obj1_eta = aln_obj1._subset_columns([0] + range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj1_eta), aln_obj1.size)
        for rec in aln_obj1_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[0] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length // 2: aln_obj1.seq_length // 2 + 5])
        aln_obj1_theta = aln_obj1._subset_columns(range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5) +
                                                  [aln_obj1.seq_length - 1])
        self.assertEqual(len(aln_obj1_theta), aln_obj1.size)
        for rec in aln_obj1_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2:
                                                                       aln_obj1.seq_length // 2 + 5] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length - 1])
        aln_obj1_iota = aln_obj1._subset_columns(range(5) + [aln_obj1.seq_length // 2] +
                                                 range(aln_obj1.seq_length - 5, aln_obj1.seq_length))
        self.assertEqual(len(aln_obj1_iota), aln_obj1.size)
        for rec in aln_obj1_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[:5] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length // 2] +
                                 aln_obj1.query_sequence[-5:])
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._subset_columns([range(5) + range(1501, 1506)])
        aln_obj2.import_alignment()
        # One position
        aln_obj2_alpha = aln_obj2._subset_columns([0])
        self.assertEqual(len(aln_obj2_alpha), aln_obj2.size)
        for rec in aln_obj2_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0])
        aln_obj2_beta = aln_obj2._subset_columns([aln_obj2.seq_length - 1])
        self.assertEqual(len(aln_obj2_beta), aln_obj2.size)
        for rec in aln_obj2_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length - 1])
        aln_obj2_gamma = aln_obj2._subset_columns([aln_obj2.seq_length // 2])
        self.assertEqual(len(aln_obj2_gamma), aln_obj2.size)
        for rec in aln_obj2_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2])
        # Single Range
        aln_obj2_delta = aln_obj2._subset_columns(range(5))
        self.assertEqual(len(aln_obj2_delta), aln_obj2.size)
        for rec in aln_obj2_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0:5])
        aln_obj2_epsilon = aln_obj2._subset_columns(range(aln_obj2.seq_length - 5, aln_obj2.seq_length))
        self.assertEqual(len(aln_obj2_epsilon), aln_obj2.size)
        for rec in aln_obj2_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[-5:])
        aln_obj2_zeta = aln_obj2._subset_columns(range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj2_zeta), aln_obj2.size)
        for rec in aln_obj2_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2:
                                                                       aln_obj2.seq_length // 2 + 5])
        # Mixed Range and Single Position
        aln_obj2_eta = aln_obj2._subset_columns([0] + range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj2_eta), aln_obj2.size)
        for rec in aln_obj2_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length // 2: aln_obj2.seq_length // 2 + 5])
        aln_obj2_theta = aln_obj2._subset_columns(range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5) +
                                                  [aln_obj2.seq_length - 1])
        self.assertEqual(len(aln_obj2_theta), aln_obj2.size)
        for rec in aln_obj2_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2:
                                                                       aln_obj2.seq_length // 2 + 5] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length - 1])
        aln_obj2_iota = aln_obj2._subset_columns(range(5) + [aln_obj2.seq_length // 2] +
                                                 range(aln_obj2.seq_length - 5, aln_obj2.seq_length))
        self.assertEqual(len(aln_obj2_iota), aln_obj2.size)
        for rec in aln_obj2_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[:5] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length // 2] +
                                 aln_obj2.query_sequence[-5:])

    def test_remove_gaps(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.remove_gaps()
        aln_obj1.import_alignment()
        aln_obj1_prime = SeqAlignment('../Test/1c17A.fa', '1c17A')
        aln_obj1_prime.import_alignment()
        aln_obj1_prime.remove_gaps()
        self.assertEqual(aln_obj1.file_name, aln_obj1_prime.file_name)
        self.assertEqual(aln_obj1.query_id, aln_obj1_prime.query_id)
        for i in range(aln_obj1.size):
            self.assertEqual(aln_obj1.alignment[i].seq, aln_obj1_prime.alignment[i].seq)
        self.assertEqual(aln_obj1.seq_order, aln_obj1_prime.seq_order)
        self.assertEqual(aln_obj1.query_sequence, aln_obj1_prime.query_sequence)
        self.assertEqual(aln_obj1.seq_length, aln_obj1_prime.seq_length)
        self.assertEqual(aln_obj1.size, aln_obj1_prime.size)
        self.assertEqual(aln_obj1.distance_matrix, aln_obj1_prime.distance_matrix)
        self.assertEqual(aln_obj1.tree_order, aln_obj1_prime.tree_order)
        self.assertEqual(aln_obj1.sequence_assignments, aln_obj1_prime.sequence_assignments)
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2.remove_gaps()
        aln_obj2.import_alignment()
        aln_obj2_prime = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        aln_obj2_prime.import_alignment()
        aln_obj2_prime.remove_gaps()
        self.assertEqual(aln_obj2.file_name, aln_obj2_prime.file_name)
        self.assertEqual(aln_obj2.query_id, aln_obj2_prime.query_id)
        for i in range(aln_obj2.size):
            self.assertEqual(len(aln_obj2_prime.alignment[i].seq), 368)
        self.assertEqual(aln_obj2.seq_order, aln_obj2_prime.seq_order)
        self.assertEqual(aln_obj2_prime.query_sequence, 'TTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHMVGMGQKDSYVGDEAQSKRGILTLKY'
                                                        'PIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPA'
                                                        'MYVAIQAVLSLYASGRTTGIVLDSGDGVTHNVPIYEGYALPHAIMRLDLAGRDLTDYLMKIL'
                                                        'TERGYSFVTTAEREIVRDIKEKLCYVALDFENEMATAASSSSLEKSYELPDGQVITIGNERF'
                                                        'RCPETLFQPSFIGMESAGIHETTYNSIMKCDIDIRKDLYANNVMSGGTTMYPGIADRMQKEI'
                                                        'TALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWITKQEYDEAGPSIVHRKCF')
        self.assertEqual(aln_obj2_prime.seq_length, 368)
        self.assertEqual(aln_obj2.size, aln_obj2_prime.size)
        self.assertEqual(aln_obj2.distance_matrix, aln_obj2_prime.distance_matrix)
        self.assertEqual(aln_obj2.tree_order, aln_obj2_prime.tree_order)
        self.assertEqual(aln_obj2.sequence_assignments, aln_obj2_prime.sequence_assignments)

    def test_compute_distance_matrix(self):
        # This is not a very good test, should come up with something else in the future, maybe compute the identity of
        # sequences separately and compare them here.
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.compute_distance_matrix()
        aln_obj1.import_alignment()
        # Compute distance matrix manually
        aln_obj1_num_mat = aln_obj1._alignment_to_num(aa_dict=aa_dict)
        value_matrix = np.zeros([aln_obj1.size, aln_obj1.size])
        for i in range(aln_obj1.size):
            check = aln_obj1_num_mat - aln_obj1_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= aln_obj1.seq_length
        value_matrix = 1 - value_matrix
        # Compute distance matrix using class method
        aln_obj1.compute_distance_matrix()
        self.assertEqual(0, np.sum(aln_obj1.distance_matrix[range(aln_obj1.size), range(aln_obj1.size)]))
        self.assertEqual(0, np.sum(value_matrix - aln_obj1.distance_matrix))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2.compute_distance_matrix()
        aln_obj2.import_alignment()
        # Compute distance matrix manually
        aln_obj2_num_mat = aln_obj2._alignment_to_num(aa_dict=aa_dict)
        value_matrix = np.zeros([aln_obj2.size, aln_obj2.size])
        for i in range(aln_obj2.size):
            check = aln_obj2_num_mat - aln_obj2_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= aln_obj2.seq_length
        value_matrix = 1 - value_matrix
        # Compute distance matrix using class method
        aln_obj2.compute_distance_matrix()
        self.assertEqual(0, np.sum(aln_obj2.distance_matrix[range(aln_obj2.size), range(aln_obj2.size)]))
        self.assertEqual(0, np.sum(value_matrix - aln_obj2.distance_matrix))

    def test_compute_effective_alignment_size(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.compute_effective_alignment_size()
        aln_obj1.import_alignment()
        aln_obj1.compute_distance_matrix()
        identity_mat = 1 - aln_obj1.distance_matrix
        effective_size = 0.0
        for i in range(aln_obj1.size):
            n_i = 0.0
            for j in range(aln_obj1.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_obj1.compute_effective_alignment_size() - effective_size), 1.0e-12)
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2.compute_effective_alignment_size()
        aln_obj2.import_alignment()
        aln_obj2.compute_distance_matrix()
        identity_mat = 1 - aln_obj2.distance_matrix
        effective_size = 0.0
        for i in range(aln_obj2.size):
            n_i = 0.0
            for j in range(aln_obj2.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_obj2.compute_effective_alignment_size() - effective_size), 1.0e-12)

    def test__agglomerative_clustering(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1._agglomerative_clustering(n_cluster=2)
        aln_obj1.import_alignment()
        aln_obj1_clusters1 = aln_obj1._agglomerative_clustering(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters1))
        self.assertTrue(1 in set(aln_obj1_clusters1))
        self.assertFalse(os.path.isdir(os.path.join(os.getcwd(), 'joblib')))
        aln_obj1_clusters2 = aln_obj1._agglomerative_clustering(n_cluster=2, cache_dir=os.path.abspath('../Test'))
        self.assertEqual(len(set(aln_obj1_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters2))
        self.assertTrue(1 in set(aln_obj1_clusters2))
        self.assertTrue(os.path.isdir(os.path.join(os.path.abspath('../Test'), 'joblib')))
        self.assertEqual(aln_obj1_clusters1, aln_obj1_clusters2)
        rmtree(os.path.join(os.path.abspath('../Test'), 'joblib'))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._agglomerative_clustering(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2_clusters1 = aln_obj2._agglomerative_clustering(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters1))
        self.assertTrue(1 in set(aln_obj2_clusters1))
        self.assertFalse(os.path.isdir(os.path.join(os.getcwd(), 'joblib')))
        aln_obj2_clusters2 = aln_obj2._agglomerative_clustering(n_cluster=2, cache_dir=os.path.abspath('../Test'))
        self.assertEqual(len(set(aln_obj2_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters2))
        self.assertTrue(1 in set(aln_obj2_clusters2))
        self.assertTrue(os.path.isdir(os.path.join(os.path.abspath('../Test'), 'joblib')))
        self.assertEqual(aln_obj2_clusters1, aln_obj2_clusters2)
        rmtree(os.path.join(os.path.abspath('../Test'), 'joblib'))

    def test_random_assignment(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1._random_assignment(n_cluster=2)
        aln_obj1.import_alignment()
        aln_obj1_clusters1 = aln_obj1._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters1))
        self.assertTrue(1 in set(aln_obj1_clusters1))
        aln_obj1_clusters2 = aln_obj1._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters2))
        self.assertTrue(1 in set(aln_obj1_clusters2))
        self.assertNotEqual(aln_obj1_clusters1, aln_obj1_clusters2)
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2_clusters1 = aln_obj2._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters1))
        self.assertTrue(1 in set(aln_obj2_clusters1))
        aln_obj2_clusters2 = aln_obj2._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters2))
        self.assertTrue(1 in set(aln_obj2_clusters2))
        self.assertNotEqual(aln_obj2_clusters1, aln_obj2_clusters2)

    def test__re_label_clusters(self):
        labels_0 = [0] * 10
        labels_1_expected = [0] * 5 + [1] * 5
        labels_1_test_1 = [1] * 5 + [0] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_1), labels_1_expected)
        labels_1_test_2 = [0] * 5 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_2), labels_1_expected)
        labels_2_expected = [0] * 3 + [1] * 2 + [2] * 5
        labels_2_test_1 = [0] * 3 + [1] * 2 + [2] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_1), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_1), labels_2_expected)
        labels_2_test_2 = [0] * 3 + [2] * 2 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_2), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_2), labels_2_expected)
        labels_2_test_3 = [1] * 3 + [0] * 2 + [2] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_3), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_3), labels_2_expected)
        labels_2_test_4 = [1] * 3 + [2] * 2 + [0] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_4), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_4), labels_2_expected)
        labels_2_test_5 = [2] * 3 + [0] * 2 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_5), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_5), labels_2_expected)
        labels_2_test_6 = [2] * 3 + [1] * 2 + [0] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_6), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_6), labels_2_expected)
        labels_3_expected = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
        labels_3_test_1 = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_1), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_1), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_1), labels_3_expected)
        labels_3_test_2 = [0] * 3 + [1] * 2 + [3] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_2), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_2), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_2), labels_3_expected)
        labels_3_test_3 = [0] * 3 + [2] * 2 + [1] * 2 + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_3), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_3), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_3), labels_3_expected)
        labels_3_test_4 = [0] * 3 + [2] * 2 + [3] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_4), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_4), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_4), labels_3_expected)
        labels_3_test_5 = [0] * 3 + [3] * 2 + [1] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_5), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_5), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_5), labels_3_expected)
        labels_3_test_6 = [0] * 3 + [3] * 2 + [2] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_6), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_6), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_6), labels_3_expected)
        labels_3_test_7 = [3] * 3 + [0] * 2 + [1] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_7), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_7), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_7), labels_3_expected)
        labels_3_test_8 = [3] * 3 + [0] * 2 + [2] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_8), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_8), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_8), labels_3_expected)
        labels_3_test_9 = [3] * 3 + [1] * 2 + [2] * 2 + [0] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_9), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_9), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_9), labels_3_expected)
        labels_3_test_10 = [3] * 3 + [1] * 2 + [0] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_10), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_10), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_10), labels_3_expected)
        labels_3_test_11 = [3] * 3 + [2] * 2 + [1] * 2 + [0] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_11), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_11), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_11), labels_3_expected)
        labels_3_test_12 = [3] * 3 + [2] * 2 + [0] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_12), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_12), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_12), labels_3_expected)
        labels_4_expected = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
        labels_4_test_1 = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_1), labels_4_expected)
        labels_4_test_2 = [0] * 3 + [1] * 2 + [2, 4] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_2), labels_4_expected)
        labels_4_test_3 = [0] * 3 + [1] * 2 + [3, 2] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_3), labels_4_expected)
        labels_4_test_4 = [0] * 3 + [1] * 2 + [3, 4] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_4), labels_4_expected)
        labels_4_test_5 = [0] * 3 + [2] * 2 + [1, 3] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_5), labels_4_expected)
        labels_4_test_6 = [0] * 3 + [2] * 2 + [1, 4] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_6), labels_4_expected)
        labels_4_test_7 = [0] * 3 + [2] * 2 + [3, 1] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_7), labels_4_expected)
        labels_4_test_8 = [0] * 3 + [2] * 2 + [3, 4] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_8), labels_4_expected)
        labels_4_test_9 = [0] * 3 + [2] * 2 + [4, 1] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_9), labels_4_expected)
        labels_4_test_10 = [0] * 3 + [2] * 2 + [4, 3] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_10), labels_4_expected)
        labels_4_test_11 = [0] * 3 + [3] * 2 + [1, 2] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_11), labels_4_expected)
        labels_4_test_12 = [0] * 3 + [3] * 2 + [1, 4] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_12), labels_4_expected)
        labels_4_test_13 = [0] * 3 + [3] * 2 + [2, 1] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_13), labels_4_expected)
        labels_4_test_14 = [0] * 3 + [3] * 2 + [2, 4] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_14), labels_4_expected)
        labels_4_test_15 = [0] * 3 + [3] * 2 + [4, 1] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_15), labels_4_expected)
        labels_4_test_16 = [0] * 3 + [3] * 2 + [4, 2] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_16), labels_4_expected)
        labels_4_test_17 = [0] * 3 + [4] * 2 + [1, 2] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_17), labels_4_expected)
        labels_4_test_18 = [0] * 3 + [4] * 2 + [1, 3] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_18), labels_4_expected)
        labels_4_test_19 = [0] * 3 + [4] * 2 + [2, 1] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_19), labels_4_expected)
        labels_4_test_20 = [0] * 3 + [4] * 2 + [2, 3] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_20), labels_4_expected)
        labels_4_test_21 = [0] * 3 + [4] * 2 + [3, 1] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_21), labels_4_expected)
        labels_4_test_22 = [0] * 3 + [4] * 2 + [3, 2] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_22), labels_4_expected)

    def test_set_tree_ordering(self):

        def check(seq_dict, curr, prev=None):
            if prev is None:
                return True
            c_prev = 0
            c_curr = 0
            while (c_prev != (prev - 1)) and (c_curr != (curr - 1)):
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    c_prev += 1
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    return False
                c_curr += 1
            return True

        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1.set_tree_ordering(tree_depth=(2, 5))
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k-1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj1.set_tree_ordering(tree_depth=clusters)
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
                            'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
        aln_obj1.set_tree_ordering()
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(1, aln_obj1.size):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2.set_tree_ordering(tree_depth=(2, 5))
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj2.set_tree_ordering(tree_depth=clusters)
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
        aln_obj2.set_tree_ordering()
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(1, aln_obj2.size):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))

    def test_visualize_tree(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1_df = aln_obj1.visualize_tree(out_dir=os.path.abspath('../Test/'))
        self.assertEqual(aln_obj1.tree_order, list(aln_obj1_df.index))
        for i in range(1, aln_obj1.size):
            for j in aln_obj1_df.index:
                self.assertIn(j,  aln_obj1.sequence_assignments[i][aln_obj1_df.loc[j, i]])
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join('../Test/',
                                                                    'query_1c17A_Sequence_Assignment.csv'))))
        os.remove(os.path.abspath(os.path.join('../Test/', 'query_1c17A_Sequence_Assignment.csv')))
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join('../Test/',
                                                                    'query_1c17A_Sequence_Assignment.eps'))))
        os.remove(os.path.abspath(os.path.join('../Test/', 'query_1c17A_Sequence_Assignment.eps')))
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2_df = aln_obj2.visualize_tree(out_dir=os.path.abspath('../Test/'))
        self.assertEqual(aln_obj2.tree_order, list(aln_obj2_df.index))
        for i in range(1, aln_obj2.size):
            for j in aln_obj2_df.index:
                self.assertIn(j,  aln_obj2.sequence_assignments[i][aln_obj2_df.loc[j, i]])
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join('../Test/',
                                                                    'query_1h1vA_Sequence_Assignment.csv'))))
        os.remove(os.path.abspath(os.path.join('../Test/', 'query_1h1vA_Sequence_Assignment.csv')))
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join('../Test/',
                                                                    'query_1h1vA_Sequence_Assignment.eps'))))
        os.remove(os.path.abspath(os.path.join('../Test/', 'query_1h1vA_Sequence_Assignment.eps')))

    def test_generate_sub_alignment(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1.set_tree_ordering(tree_depth=(2, 5))
        for k in aln_obj1.sequence_assignments:
            for c in aln_obj1.sequence_assignments[k]:
                aln_obj1_sub = aln_obj1.generate_sub_alignment(aln_obj1.sequence_assignments[k][c])
                self.assertEqual(aln_obj1.file_name, aln_obj1_sub.file_name)
                self.assertEqual(aln_obj1.query_id, aln_obj1_sub.query_id)
                self.assertEqual(aln_obj1.query_sequence, aln_obj1_sub.query_sequence)
                self.assertIsNone(aln_obj1_sub.distance_matrix)
                self.assertIsNone(aln_obj1_sub.sequence_assignments)
                self.assertEqual(aln_obj1_sub.size, len(aln_obj1.sequence_assignments[k][c]))
                self.assertEqual(aln_obj1_sub.seq_order, [x for x in aln_obj1.seq_order
                                                          if x in aln_obj1.sequence_assignments[k][c]])
                self.assertEqual(aln_obj1_sub.tree_order, [x for x in aln_obj1.tree_order
                                                           if (x in aln_obj1.sequence_assignments[k][c])])
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2.set_tree_ordering(tree_depth=(2, 5))
        for k in aln_obj2.sequence_assignments:
            for c in aln_obj2.sequence_assignments[k]:
                aln_obj2_sub = aln_obj2.generate_sub_alignment(aln_obj2.sequence_assignments[k][c])
                self.assertEqual(aln_obj2.file_name, aln_obj2_sub.file_name)
                self.assertEqual(aln_obj2.query_id, aln_obj2_sub.query_id)
                self.assertEqual(aln_obj2.query_sequence, aln_obj2_sub.query_sequence)
                self.assertIsNone(aln_obj2_sub.distance_matrix)
                self.assertIsNone(aln_obj2_sub.sequence_assignments)
                self.assertEqual(aln_obj2_sub.size, len(aln_obj2.sequence_assignments[k][c]))
                self.assertEqual(aln_obj2_sub.seq_order, [x for x in aln_obj2.seq_order
                                                          if x in aln_obj2.sequence_assignments[k][c]])
                self.assertEqual(aln_obj2_sub.tree_order, [x for x in aln_obj2.tree_order
                                                           if x in aln_obj2.sequence_assignments[k][c]])

    def test_get_branch_cluster(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1.set_tree_ordering(tree_depth=(2, 5))
        for k in aln_obj1.sequence_assignments:
            for c in aln_obj1.sequence_assignments[k]:
                aln_obj1_sub = aln_obj1.get_branch_cluster(k, c)
                aln_obj1_sub_prime = aln_obj1.generate_sub_alignment(aln_obj1.sequence_assignments[k][c])
                self.assertEqual(aln_obj1_sub.file_name, aln_obj1_sub_prime.file_name)
                self.assertEqual(aln_obj1_sub.query_id, aln_obj1_sub_prime.query_id)
                self.assertEqual(aln_obj1.query_sequence, aln_obj1_sub.query_sequence)
                self.assertEqual(aln_obj1_sub.distance_matrix, aln_obj1_sub_prime.distance_matrix)
                self.assertEqual(aln_obj1_sub.sequence_assignments, aln_obj1_sub_prime.sequence_assignments)
                self.assertEqual(aln_obj1_sub.size, aln_obj1_sub_prime.size)
                self.assertEqual(aln_obj1_sub.seq_order, aln_obj1_sub_prime.seq_order)
                self.assertEqual(aln_obj1_sub.tree_order, aln_obj1_sub_prime.tree_order)
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2.set_tree_ordering(tree_depth=(2, 5))
        for k in aln_obj2.sequence_assignments:
            for c in aln_obj2.sequence_assignments[k]:
                aln_obj2_sub = aln_obj2.get_branch_cluster(k, c)
                aln_obj2_sub_prime = aln_obj2.generate_sub_alignment(aln_obj2.sequence_assignments[k][c])
                self.assertEqual(aln_obj2_sub.file_name, aln_obj2_sub_prime.file_name)
                self.assertEqual(aln_obj2_sub.query_id, aln_obj2_sub_prime.query_id)
                self.assertEqual(aln_obj2.query_sequence, aln_obj2_sub.query_sequence)
                self.assertEqual(aln_obj2_sub.distance_matrix, aln_obj2_sub_prime.distance_matrix)
                self.assertEqual(aln_obj2_sub.sequence_assignments, aln_obj2_sub_prime.sequence_assignments)
                self.assertEqual(aln_obj2_sub.size, aln_obj2_sub_prime.size)
                self.assertEqual(aln_obj2_sub.seq_order, aln_obj2_sub_prime.seq_order)
                self.assertEqual(aln_obj2_sub.tree_order, aln_obj2_sub_prime.tree_order)

    def test_generate_positional_sub_alignment(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        for i in range(aln_obj1.size - 1):
            aln_obj1_sub = aln_obj1.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_obj1.file_name, aln_obj1_sub.file_name)
            self.assertEqual(aln_obj1.query_id, aln_obj1_sub.query_id)
            self.assertEqual(aln_obj1_sub.query_sequence, aln_obj1.query_sequence[i] + aln_obj1.query_sequence[i + 1])
            self.assertIsNone(aln_obj1_sub.distance_matrix)
            self.assertIsNone(aln_obj1_sub.sequence_assignments)
            self.assertEqual(aln_obj1.size, aln_obj1_sub.size)
            self.assertEqual(aln_obj1.seq_order, aln_obj1_sub.seq_order)
            self.assertEqual(aln_obj1.tree_order, aln_obj1_sub.tree_order)
            for j in range(aln_obj1.size):
                self.assertEqual(aln_obj1_sub.alignment[j].seq,
                                 aln_obj1.alignment[j].seq[i] + aln_obj1.alignment[j].seq[i + 1])
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        for i in range(aln_obj2.size - 1):
            aln_obj2_sub = aln_obj2.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_obj2.file_name, aln_obj2_sub.file_name)
            self.assertEqual(aln_obj2.query_id, aln_obj2_sub.query_id)
            self.assertEqual(aln_obj2_sub.query_sequence, aln_obj2.query_sequence[i] + aln_obj2.query_sequence[i + 1])
            self.assertIsNone(aln_obj2_sub.distance_matrix)
            self.assertIsNone(aln_obj2_sub.sequence_assignments)
            self.assertEqual(aln_obj2.size, aln_obj2_sub.size)
            self.assertEqual(aln_obj2.seq_order, aln_obj2_sub.seq_order)
            self.assertEqual(aln_obj2.tree_order, aln_obj2_sub.tree_order)
            for j in range(aln_obj2.size):
                self.assertEqual(aln_obj2_sub.alignment[j].seq,
                                 aln_obj2.alignment[j].seq[i] + aln_obj2.alignment[j].seq[i + 1])

    def test_determine_usable_positions(self):
        aln_obj1 = SeqAlignment('../Test/1c17A.fa', '1c17A')
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        pos1, evidence1 = aln_obj1.determine_usable_positions(ratio=0.5)
        usable_pos = []
        for i in range(aln_obj1.seq_length):
            count = 0
            for j in range(aln_obj1.size):
                if aln_obj1.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_obj1.size / 2):
                usable_pos.append(i)
            self.assertEqual(evidence1[i], count)
        self.assertEqual(list(pos1), usable_pos)
        aln_obj2 = SeqAlignment('../Test/1h1vA.fa', '1h1vA')
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        pos2, evidence2 = aln_obj2.determine_usable_positions(ratio=0.5)
        usable_pos2 = []
        for i in range(aln_obj2.seq_length):
            count = 0
            for j in range(aln_obj2.size):
                if aln_obj2.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_obj2.size / 2):
                usable_pos2.append(i)
            self.assertEqual(evidence2[i], count)
        self.assertEqual(list(pos2), usable_pos2)

    # def test_identify_comparable_sequences(self):
    #     self.fail()
    #
    # def test_heatmap_plot(self):
    #     self.fail()
    #
    # def test_alignment_to_num(self):
    #     self.fail()
    #
