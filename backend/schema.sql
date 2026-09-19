PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 username TEXT NOT NULL UNIQUE,
 password_hash TEXT NOT NULL,
 display_name TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','disabled')),
 created_at TEXT NOT NULL,
 last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS learning_materials (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER NOT NULL REFERENCES users(id),
 original_filename TEXT NOT NULL,
 file_type TEXT NOT NULL,
 file_size INTEGER NOT NULL CHECK(file_size >= 0),
 file_path TEXT NOT NULL,
 content_hash TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'uploaded' CHECK(status IN ('uploaded','processing','ready','failed')),
 processing_config_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(processing_config_json)),
 vector_store_path TEXT,
 chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
 knowledge_points_json TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(knowledge_points_json)),
 error_message TEXT,
 created_at TEXT NOT NULL,
 processed_at TEXT,
 UNIQUE(id,user_id)
);
CREATE INDEX IF NOT EXISTS materials_owner ON learning_materials(user_id,created_at);
CREATE INDEX IF NOT EXISTS materials_hash ON learning_materials(user_id,content_hash);
CREATE TABLE IF NOT EXISTS questions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER NOT NULL REFERENCES users(id),
 material_id INTEGER NOT NULL,
 primary_knowledge_point_id INTEGER NOT NULL CHECK(primary_knowledge_point_id > 0),
 secondary_knowledge_point_ids_json TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(secondary_knowledge_point_ids_json)),
 question_type TEXT NOT NULL CHECK(question_type IN ('single_choice','multiple_choice','fill_blank','short_answer')),
 difficulty_level INTEGER NOT NULL CHECK(difficulty_level BETWEEN 1 AND 5),
 skill_type TEXT NOT NULL,
 question_summary TEXT NOT NULL,
 source_refs_json TEXT NOT NULL CHECK(json_valid(source_refs_json)),
 generation_context_json TEXT NOT NULL CHECK(json_valid(generation_context_json)),
 model_name TEXT NOT NULL,
 prompt_version TEXT NOT NULL,
 policy_version TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'ready' CHECK(status IN ('ready','expired','invalid')),
 content_expires_at TEXT,
 content_snapshot_json TEXT CHECK(content_snapshot_json IS NULL OR json_valid(content_snapshot_json)),
 created_at TEXT NOT NULL,
 UNIQUE(id,user_id),
 FOREIGN KEY(material_id,user_id) REFERENCES learning_materials(id,user_id)
);
CREATE TRIGGER IF NOT EXISTS immutable_question_snapshot
BEFORE UPDATE OF content_snapshot_json ON questions
WHEN OLD.content_snapshot_json IS NOT NULL AND NEW.content_snapshot_json IS NOT OLD.content_snapshot_json
BEGIN SELECT RAISE(ABORT,'Saved question content is immutable'); END;
CREATE TRIGGER IF NOT EXISTS question_knowledge_points
BEFORE INSERT ON questions
WHEN NOT EXISTS (SELECT 1 FROM learning_materials m,json_each(m.knowledge_points_json) k
 WHERE m.id=NEW.material_id AND json_extract(k.value,'$.id')=NEW.primary_knowledge_point_id)
 OR EXISTS (SELECT 1 FROM json_each(NEW.secondary_knowledge_point_ids_json) s WHERE NOT EXISTS
 (SELECT 1 FROM learning_materials m,json_each(m.knowledge_points_json) k
 WHERE m.id=NEW.material_id AND json_extract(k.value,'$.id')=s.value))
BEGIN SELECT RAISE(ABORT,'Unknown knowledge point'); END;
CREATE TABLE IF NOT EXISTS question_attempts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER NOT NULL REFERENCES users(id),
 question_id INTEGER NOT NULL,
 attempt_no INTEGER NOT NULL CHECK(attempt_no > 0),
 status TEXT NOT NULL DEFAULT 'in_progress' CHECK(status IN ('in_progress','pending_grading','completed','abandoned','expired','grading_failed')),
 started_at TEXT NOT NULL,
 submitted_at TEXT,
 last_activity_at TEXT NOT NULL,
 is_manually_paused INTEGER NOT NULL DEFAULT 0 CHECK(is_manually_paused IN (0,1)),
 is_page_hidden INTEGER NOT NULL DEFAULT 0 CHECK(is_page_hidden IN (0,1)),
 active_duration_ms INTEGER NOT NULL DEFAULT 0 CHECK(active_duration_ms >= 0),
 timing_updated_at TEXT NOT NULL,
 progress_seq INTEGER NOT NULL DEFAULT 0 CHECK(progress_seq >= 0),
 answer_json TEXT CHECK(answer_json IS NULL OR json_valid(answer_json)),
 answer_summary TEXT,
 is_correct INTEGER CHECK(is_correct IN (0,1)),
 score REAL CHECK(score BETWEEN 0 AND 1),
 feedback_summary TEXT,
 answer_revealed_at TEXT,
 grading_method TEXT,
 grading_version TEXT,
 grading_confidence TEXT NOT NULL DEFAULT 'unknown' CHECK(grading_confidence IN ('high','medium','low','unknown')),
 submission_key TEXT,
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 UNIQUE(user_id,question_id,attempt_no),
 UNIQUE(user_id,submission_key),
 FOREIGN KEY(question_id,user_id) REFERENCES questions(id,user_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_attempt ON question_attempts(user_id,question_id) WHERE status='in_progress';
CREATE INDEX IF NOT EXISTS attempts_history ON question_attempts(user_id,started_at DESC,id DESC);
CREATE TABLE IF NOT EXISTS hint_records (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 attempt_id INTEGER NOT NULL REFERENCES question_attempts(id),
 sequence_no INTEGER NOT NULL CHECK(sequence_no > 0),
 hint_type TEXT NOT NULL CHECK(hint_type IN ('concept','approach','step','answer')),
 hint_level INTEGER NOT NULL CHECK(hint_level BETWEEN 1 AND 3),
 request_text TEXT,
 hint_summary TEXT,
 status TEXT NOT NULL CHECK(status IN ('requested','generated','viewed','failed')),
 requested_at TEXT NOT NULL,
 delivered_at TEXT,
 viewed_at TEXT,
 active_elapsed_ms INTEGER NOT NULL CHECK(active_elapsed_ms >= 0),
 request_key TEXT NOT NULL,
 model_name TEXT,
 prompt_version TEXT,
 UNIQUE(attempt_id,sequence_no),
 UNIQUE(attempt_id,request_key)
);
CREATE TABLE IF NOT EXISTS error_records (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 attempt_id INTEGER NOT NULL REFERENCES question_attempts(id),
 knowledge_point_id INTEGER NOT NULL CHECK(knowledge_point_id > 0),
 error_type TEXT NOT NULL CHECK(error_type IN ('concept_missing','confusion','method','calculation','reading','expression','unknown')),
 description TEXT NOT NULL,
 evidence_summary TEXT NOT NULL,
 source TEXT NOT NULL CHECK(source IN ('user','rule','model')),
 confidence TEXT NOT NULL CHECK(confidence IN ('high','medium','low','unknown')),
 review_status TEXT NOT NULL DEFAULT 'pending' CHECK(review_status IN ('pending','confirmed','rejected')),
 suggestion TEXT,
 analyzer_version TEXT NOT NULL DEFAULT 'v1',
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS error_knowledge_point
BEFORE INSERT ON error_records
WHEN NOT EXISTS (SELECT 1 FROM question_attempts a JOIN questions q ON q.id=a.question_id
 JOIN learning_materials m ON m.id=q.material_id JOIN json_each(m.knowledge_points_json) k
 WHERE a.id=NEW.attempt_id AND json_extract(k.value,'$.id')=NEW.knowledge_point_id)
BEGIN SELECT RAISE(ABORT,'Unknown error knowledge point'); END;
