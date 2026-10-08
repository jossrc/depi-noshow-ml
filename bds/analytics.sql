create type appointment_outcome_enum as enum ('ATTENDED_COMPLETED', 'NO_SHOW', 'CANCELLED', 'RESCHEDULED_HISTORY', 'IN_PROGRESS', 'FUTURE', 'BLOCKED', 'DRAFT', 'IGNORED', 'UNKNOWN');

alter type appointment_outcome_enum owner to admin;

create type appointment_detail_enum as enum ('ATTENDED', 'NO_SHOW_CANDIDATE', 'NO_SHOW_REVIEW_SYSTEM', 'CANCELLED_OR_UNCLASSIFIED', 'FUTURE_CANCELLED', 'FUTURE_PENDING', 'SESSION_IN_PROGRESS', 'PAST_UNRESOLVED', 'REVIEW_CONFLICT', 'EXCLUDED');

alter type appointment_detail_enum owner to admin;

create type appointment_tracking_origin_enum as enum ('UID_MINUS_ONE', 'USER_POSITIVE', 'ZERO_OR_NULL');

alter type appointment_tracking_origin_enum owner to admin;

create type appointment_exclusion_reason_enum as enum ('BLOCKED_SLOT', 'RESCHEDULE_HISTORY', 'DRAFT_TAG_16', 'CLINIC_8_LEGACY_RULE', 'OTHER_EXCLUSION');

alter type appointment_exclusion_reason_enum owner to admin;

create type appointment_event_type_enum as enum ('CREATED', 'CONFIRMED', 'ARRIVED_AT_CLINIC', 'RESCHEDULED', 'CANCELLED', 'NO_SHOW', 'SESSION_OPENED', 'SESSION_CLOSED', 'TICKET_CLOSED');

alter type appointment_event_type_enum owner to admin;

create type appointment_service_phase_enum as enum ('SCHEDULED', 'PERFORMED', 'BILLED');

alter type appointment_service_phase_enum owner to admin;

create table appointment_outcomes
(
    appointment_id           bigint                                             not null
        primary key,
    client_id                bigint                                             not null,
    clinic_id                integer                                            not null,
    appointment_at           timestamp with time zone                           not null,
    created_at_source        timestamp with time zone,
    tracking_at_source       timestamp with time zone,
    tracking_user_id         bigint,
    diary_not_attended_raw   smallint                 default 0                 not null,
    diary_delayed_raw        smallint                 default 0                 not null,
    diary_locked_raw         smallint                 default 0                 not null,
    diary_laser_id           bigint,
    ticket_id                bigint,
    ticket_closed_raw        smallint,
    outcome                  analytics.appointment_outcome_enum                 not null,
    outcome_resolved_at      timestamp with time zone,
    outcome_rule_version     text                                               not null,
    training_eligible        boolean                  default false             not null,
    exclusion_reason         analytics.appointment_exclusion_reason_enum,
    source_batch_id          bigint,
    computed_at              timestamp with time zone default CURRENT_TIMESTAMP not null,
    classification_detail    analytics.appointment_detail_enum,
    is_no_show_candidate     boolean                  default false             not null,
    tracking_origin          analytics.appointment_tracking_origin_enum,
    requires_review          boolean                  default false             not null,
    classification_cutoff_at timestamp with time zone
);

alter table appointment_outcomes
    owner to admin;

create index idx_outcomes_client_at
    on appointment_outcomes (client_id, appointment_at);

create index idx_outcomes_outcome
    on appointment_outcomes (outcome, appointment_at);

create table appointment_events
(
    event_id        bigint generated always as identity
        primary key,
    appointment_id  bigint                                             not null,
    client_id       bigint                                             not null,
    clinic_id       integer,
    event_type      analytics.appointment_event_type_enum              not null,
    event_at        timestamp with time zone,
    source_table    text                                               not null,
    source_row_id   bigint,
    rule_version    text,
    metadata        jsonb                    default '{}'::jsonb       not null,
    source_batch_id bigint,
    computed_at     timestamp with time zone default CURRENT_TIMESTAMP not null
);

alter table appointment_events
    owner to admin;

create unique index uq_appointment_events_source_type
    on appointment_events (source_table, source_row_id, event_type);

create index idx_appointment_events_client
    on appointment_events (client_id, event_at)
    where (event_at IS NOT NULL);

create index idx_appointment_events_app
    on appointment_events (appointment_id, event_at)
    where (event_at IS NOT NULL);

create table client_history_current
(
    client_id                        bigint                                             not null
        primary key,
    as_of_at                         timestamp with time zone                           not null,
    past_appointments                integer                  default 0                 not null,
    attended_completed               integer                  default 0                 not null,
    no_shows_candidate               integer                  default 0                 not null,
    no_shows_uid_minus_one           integer                  default 0                 not null,
    cancellation_marks_past          integer                  default 0                 not null,
    past_unresolved                  integer                  default 0                 not null,
    future_cancelled                 integer                  default 0                 not null,
    future_pending                   integer                  default 0                 not null,
    reschedule_events_linked         integer                  default 0                 not null,
    reschedule_appointments_distinct integer                  default 0                 not null,
    reschedule_events_unlinked       integer                  default 0                 not null,
    no_show_rate_on_resolved         numeric(10, 6),
    cancellation_mark_rate_on_past   numeric(10, 6),
    last_scheduled_appointment_at    timestamp with time zone,
    last_attended_at                 timestamp with time zone,
    last_no_show_at                  timestamp with time zone,
    last_cancelled_appointment_at    timestamp with time zone,
    days_since_last_attended         integer,
    days_since_last_no_show          integer,
    last_past_outcome                analytics.appointment_outcome_enum,
    feature_version                  text                                               not null,
    safe_for_historical_training     boolean                  default false             not null
        constraint client_history_current_safe_for_historical_training_check
            check (safe_for_historical_training = false),
    computed_at                      timestamp with time zone default CURRENT_TIMESTAMP not null
);

alter table client_history_current
    owner to admin;

create table appointment_services
(
    appointment_id         bigint                                             not null,
    service_phase          analytics.appointment_service_phase_enum           not null,
    source_detail_id       bigint                                             not null,
    product_id             bigint,
    product_parent_id      bigint,
    voucher_id             bigint,
    units                  numeric(14, 3),
    product_type           smallint,
    product_zone_code      varchar(16),
    requires_doctor        boolean,
    metadata               jsonb                    default '{}'::jsonb       not null,
    computed_at            timestamp with time zone default CURRENT_TIMESTAMP not null,
    pack_id                bigint,
    planned_session_number smallint,
    product_family_id      bigint,
    is_medical_evaluation  boolean,
    primary key (appointment_id, service_phase, source_detail_id)
);

alter table appointment_services
    owner to admin;

create table product_catalog_resolution
(
    product_id              bigint                                             not null
        primary key,
    product_type            smallint,
    product_parent_id       bigint,
    voucher_zone_product_id bigint,
    body_area_product_id    bigint,
    product_family_id       bigint,
    product_zone_code       varchar(32),
    resolution_method       varchar(24)                                        not null
        constraint product_catalog_resolution_resolution_method_check
            check ((resolution_method)::text = ANY
                   ((ARRAY ['SERVICE_PARENT'::character varying, 'VOUCHER_ZONE_PARENT'::character varying, 'MISSING_ZONE'::character varying, 'MISSING_PARENT'::character varying, 'NOT_APPLICABLE'::character varying])::text[])),
    updated_at              timestamp with time zone default CURRENT_TIMESTAMP not null
);

alter table product_catalog_resolution
    owner to admin;

create table voucher_features_current
(
    client_id                    bigint                                             not null,
    body_area_product_id         bigint                                             not null,
    voucher_count                integer                                            not null,
    available_units              bigint                                             not null,
    current_unexpired_vouchers   integer                                            not null,
    snapshot_at                  timestamp with time zone default CURRENT_TIMESTAMP not null,
    safe_for_historical_training boolean                  default false             not null
        constraint voucher_features_current_safe_for_historical_training_check
            check (safe_for_historical_training = false),
    primary key (client_id, body_area_product_id)
);

alter table voucher_features_current
    owner to admin;

create table ml_history_cumulative_v1
(
    appointment_id                 bigint                                             not null
        primary key,
    client_id                      bigint                                             not null,
    known_at                       timestamp with time zone                           not null,
    event_kind                     smallint                                           not null
        constraint ml_history_cumulative_v1_event_kind_check
            check (event_kind = ANY (ARRAY [0, 1])),
    prior_including_event_attended integer                                            not null,
    prior_including_event_no_show  integer                                            not null,
    last_attended_at               timestamp with time zone,
    last_no_show_at                timestamp with time zone,
    updated_at                     timestamp with time zone default CURRENT_TIMESTAMP not null
);

alter table ml_history_cumulative_v1
    owner to admin;

create index idx_ml_history_client_known_at
    on ml_history_cumulative_v1 (client_id asc, known_at desc, appointment_id desc);

create table appointment_training_dataset_v1
(
    appointment_id               bigint                                                       not null
        primary key,
    client_id                    bigint                                                       not null,
    prediction_at                timestamp with time zone                                     not null,
    appointment_at               timestamp with time zone                                     not null,
    target                       smallint                                                     not null
        constraint appointment_training_dataset_v1_target_check
            check (target = ANY (ARRAY [0, 1])),
    no_show_uid_minus_one        boolean                                                      not null,
    clinic_id                    integer                                                      not null,
    client_sex                   varchar(1),
    age_at_booking               smallint,
    booking_lead_days            double precision                                             not null,
    appointment_month            smallint,
    appointment_weekday          smallint,
    appointment_hour             smallint,
    duration_minutes             integer,
    is_fwa                       boolean,
    scheduled_service_lines      integer                                                      not null,
    distinct_body_areas          integer                                                      not null,
    single_body_area_id          bigint,
    has_medical_evaluation       boolean,
    has_type4_service            boolean,
    previous_attended            integer                                                      not null,
    previous_no_show             integer                                                      not null,
    previous_no_show_rate        double precision,
    days_since_previous_attended double precision,
    days_since_previous_no_show  double precision,
    feature_version              text                     default 'BOOKING_BASELINE_V1'::text not null,
    built_at                     timestamp with time zone default CURRENT_TIMESTAMP           not null
);

alter table appointment_training_dataset_v1
    owner to admin;

create view v_appointment_outcomes_v1
            (appointment_id, client_id, clinic_id, appointment_at, created_at, tracking_at, tracking_user_id,
             has_cancel_mark, has_session, has_closed_ticket, meets_no_show_rule, outcome, classification_rule_version)
as
WITH base AS (SELECT dg."DiaryGID"                                            AS appointment_id,
                     dg."DiaryGClientID"                                      AS client_id,
                     dg."DiaryGClinicID"                                      AS clinic_id,
                     dg."DiaryGDate" + dg."DiaryGStart"                       AS appointment_at,
                     dg."DiaryGCreated"                                       AS created_at,
                     dg."DiaryGTrackingStamp"                                 AS tracking_at,
                     dg."DiaryGTrackingUID"                                   AS tracking_user_id,
                     dg."DiaryGNotAttended" = '-1'::integer                   AS has_cancel_mark,
                     lg."LaserGID" IS NOT NULL                                AS has_session,
                     tg."TicketGClosed" = '-1'::integer                       AS has_closed_ticket,
                     dg."DiaryGLocked" = '-1'::integer OR dg."DiaryGDelayed" = '-1'::integer OR
                     dg."DiaryGClinicID" = 8 OR (('|'::text ||
                                                  regexp_replace(COALESCE(dg."DiaryGTags", ''::character varying)::text,
                                                                 '[[:space:]]+'::text, ''::text, 'g'::text)) ||
                                                 '|'::text) ~~ '%|16|%'::text AS is_excluded
              FROM raw_flow.diary_gen dg
                       LEFT JOIN raw_flow.laser_gen lg ON lg."LaserGID" = dg."DiaryGLaserGID"
                       LEFT JOIN raw_flow.tickets_gen tg ON tg."TicketGID" = lg."LaserGTicketGID"),
     flags AS (SELECT base.appointment_id,
                      base.client_id,
                      base.clinic_id,
                      base.appointment_at,
                      base.created_at,
                      base.tracking_at,
                      base.tracking_user_id,
                      base.has_cancel_mark,
                      base.has_session,
                      base.has_closed_ticket,
                      base.is_excluded,
                      base.tracking_user_id <> 0 AND base.tracking_at IS NOT NULL AND
                      base.appointment_at >= (base.tracking_at - '03:00:00'::interval) AND
                      base.appointment_at <= base.tracking_at AND
                      (base.created_at + '03:00:00'::interval) < base.tracking_at AND (base.tracking_user_id <> ALL
                                                                                       (ARRAY [645::bigint, 646::bigint, 740::bigint, 802::bigint, 803::bigint])) AS meets_no_show_rule
               FROM base)
SELECT appointment_id,
       client_id,
       clinic_id,
       appointment_at,
       created_at,
       tracking_at,
       tracking_user_id,
       has_cancel_mark,
       has_session,
       has_closed_ticket,
       meets_no_show_rule,
       CASE
           WHEN is_excluded THEN 'EXCLUDED'::text
           WHEN has_cancel_mark AND has_closed_ticket THEN 'REVIEW_CONFLICT'::text
           WHEN appointment_at > (CURRENT_TIMESTAMP AT TIME ZONE 'America/Lima'::text) AND has_cancel_mark
               THEN 'FUTURE_CANCELLED'::text
           WHEN appointment_at > (CURRENT_TIMESTAMP AT TIME ZONE 'America/Lima'::text) THEN 'FUTURE_PENDING'::text
           WHEN has_cancel_mark AND meets_no_show_rule AND tracking_user_id = '-1'::integer THEN 'NO_SHOW_REVIEW_SYSTEM'::text
           WHEN has_cancel_mark AND meets_no_show_rule AND tracking_user_id > 0 THEN 'NO_SHOW_CANDIDATE'::text
           WHEN has_cancel_mark THEN 'CANCELLED_OR_UNCLASSIFIED'::text
           WHEN has_closed_ticket THEN 'ATTENDED'::text
           WHEN has_session THEN 'SESSION_IN_PROGRESS'::text
           ELSE 'PAST_UNRESOLVED'::text
           END                 AS outcome,
       'FLOW_NO_SHOW_V1'::text AS classification_rule_version
FROM flags;

alter table v_appointment_outcomes_v1
    owner to admin;

