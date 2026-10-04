export type Patient = { id: string; display_name: string; disease_pack: string };

export type Source = {
  document_id: string;
  document_name: string;
  page_number: number;
  source_snippet: string;
};

export type CancerState = {
  diagnosis: string | null;
  subtype: string | null;
  receptors: Record<string, string | null>;
  current_treatment: { name: string; date: string | null; active: boolean; fact_id: string } | null;
  disease_sites: string[];
  latest_scan: { assessment: string; date: string | null; fact_id: string } | null;
  genomics: { alteration: string; date: string | null; fact_id: string }[];
  missing_information: string[];
};

export type StateResponse = {
  snapshot_id: string | null;
  state_hash: string | null;
  state: CancerState;
  sources: Record<string, Source>;
};

export type DocumentRecord = {
  id: string;
  filename: string;
  document_type: string | null;
  status: string;
  created_at: string;
  processing_error?: string | null;
};

export type CandidateFact = {
  id: string;
  document_id: string;
  document_name: string;
  fact_type: string;
  value: string;
  effective_date: string | null;
  confidence: number;
  source_page: number;
  source_snippet: string;
  review_reason: string | null;
  status: string;
};

export type TimelineEvent = {
  id: string;
  event_type: string;
  event_date: string | null;
  title: string;
  summary: string;
  source: Source | null;
};

export type ForecastSupport = {
  status?: 'STANDARD' | 'LIMITED' | 'UNAVAILABLE' | string;
  reason_codes?: string[];
  explanations?: string[];
};

export type ForecastHorizon = {
  pre_pfs: number;
  selected_pfs: number;
  post_pfs: number;
};

export type Forecast = {
  id?: string;
  status?: string;
  patient_id?: string;
  state_hash?: string;
  run_status?: string;
  research_status?: string;
  support?: ForecastSupport;
  model?: {
    checkpoint?: string;
    selected_rule?: string;
    locked_alpha?: number;
    deploy_sha256?: string;
    temporal_encoder_sha256?: string;
    adapter_version?: string;
  };
  current_scan?: { fact_id?: string | null; date?: string | null };
  curves?: {
    months: number[];
    pre_pfs: number[];
    selected_pfs: number[];
    post_pfs: number[];
  } | null;
  horizons?: Record<string, ForecastHorizon> | null;
  prepared_hashes?: Record<string, string>;
  created_at?: string;
  reused?: boolean;
};

// === ONCOTWIN CP4 PATIENT INTELLIGENCE TYPES ===
export type GroundingRef = {
  key: string;
  kind: 'record' | 'literature' | 'trial';
  label: string;
  document_id?: string | null;
  document_name?: string | null;
  page_number?: number | null;
  source_snippet?: string | null;
  source_id?: string | null;
  url?: string | null;
};

export type IntelligenceArtifact = {
  id?: string | null;
  artifact_type: string;
  status: string;
  cache_hit?: boolean;
  patient_id?: string | null;
  document_id?: string | null;
  state_hash?: string | null;
  provider?: string | null;
  model_name?: string | null;
  prompt_version?: string | null;
  generated_at?: string | null;
  content: Record<string, any>;
  grounding?: GroundingRef[];
};

export type IntelligenceBundle = {
  patient_id: string;
  state_hash: string;
  what_changed?: IntelligenceArtifact | null;
  visit_questions?: IntelligenceArtifact | null;
  visit_brief?: IntelligenceArtifact | null;
  evidence?: IntelligenceArtifact | null;
  export?: IntelligenceArtifact | null;
};

export type EvidenceBundle = {
  artifact_id?: string;
  cache_hit?: boolean;
  status: string;
  mode: string;
  retrieval_version?: string;
  queries?: Record<string, string>;
  literature: Array<Record<string, any>>;
  trials: Array<Record<string, any>>;
  note?: string | null;
  trial_disclaimer?: string;
};
// === END ONCOTWIN CP4 PATIENT INTELLIGENCE TYPES ===

// === ONCOTWIN CP4.5 ASK ONCOTWIN TYPES ===
export type AssistantTurn = {
  role: 'user' | 'assistant';
  content: string;
};

export type AssistantGrounding = GroundingRef;

export type AssistantBootstrap = {
  stage: 'baseline' | 'after_progression' | 'after_esr1';
  mode: 'fixture' | 'llm';
  greeting: string;
  suggested_questions: string[];
};

export type AssistantAnswer = {
  stage: 'baseline' | 'after_progression' | 'after_esr1';
  mode: 'fixture' | 'llm';
  matched_question: boolean;
  answer: string;
  suggested_questions: string[];
  suggested_followups: string[];
  grounding: AssistantGrounding[];
  forecast_refs: Record<string, any>;
  provider?: string | null;
  model_name?: string | null;
  uncertainty_note?: string | null;
};
// === END ONCOTWIN CP4.5 ASK ONCOTWIN TYPES ===

