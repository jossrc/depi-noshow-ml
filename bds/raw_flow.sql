create table ingestion_batches
(
    batch_id        bigint generated always as identity
        primary key,
    source_type     text                                             not null
        constraint ingestion_batches_source_type_check
            check (source_type = ANY (ARRAY ['DIRECT_MYSQL'::text, 'CSV'::text])),
    source_name     text,
    dataset_date    date,
    started_at      timestamp with time zone default now()           not null,
    finished_at     timestamp with time zone,
    status          text                     default 'RUNNING'::text not null
        constraint ingestion_batches_status_check
            check (status = ANY (ARRAY ['RUNNING'::text, 'COMPLETED'::text, 'FAILED'::text, 'PARTIAL'::text])),
    row_counts      jsonb                    default '{}'::jsonb     not null,
    checksum_sha256 text,
    notes           text,
    error_message   text
);

alter table ingestion_batches
    owner to admin;

create table x_config_clinics
(
    "ClinicID"             integer                                                not null
        primary key,
    "ClinicName"           varchar(128)             default ''::character varying not null,
    "ClinicCommercialName" varchar(128)             default ''::character varying not null,
    "ClinicTimeStart"      time,
    "ClinicTimeEnd"        time,
    "ClinicSAT"            smallint                 default 0                     not null,
    "ClinicSATTimeStart"   time,
    "ClinicSATTimeEnd"     time,
    "ClinicSUN"            smallint                 default 0                     not null,
    "ClinicAppConcurrent"  smallint                 default 0                     not null,
    "ClinicCity"           varchar(64)              default ''::character varying not null,
    "ClinicProvinceID"     bigint                   default 0                     not null,
    "ClinicRegionID"       smallint                 default 0                     not null,
    "ClinicCountryID"      smallint                 default 0                     not null,
    "ClinicTimeZoneID"     integer                  default 0                     not null,
    "ClinicProductGID"     bigint                   default 0                     not null,
    "ClinicStatsDisabled"  smallint                 default 0                     not null,
    "ClinicDisabled"       smallint                 default 0                     not null,
    "ClinicReadOnly"       smallint                 default 0                     not null,
    "ClinicTest"           smallint                 default 0                     not null,
    "ClinicStopped"        smallint                 default 0                     not null,
    "ClinicStandBy"        smallint                 default 0                     not null,
    "ClinicDateNew"        date,
    "ClinicDateDisabled"   date,
    "ClinicStamp"          timestamp,
    _ingested_at           timestamp with time zone default now()                 not null,
    _ingestion_batch_id    bigint
        references ingestion_batches
);

alter table x_config_clinics
    owner to admin;

create table x_config_products_det
(
    "ProductID"                    bigint                                                 not null
        primary key,
    "ProductParentID"              bigint                   default 0                     not null,
    "ProductSourceID"              bigint                   default 0                     not null,
    "ProductGID"                   bigint                   default 0                     not null,
    "ProductFamilyID"              bigint                   default 0                     not null,
    "ProductZoneClassID"           bigint,
    "ProductZoneCode"              varchar(16),
    "ProductDesc"                  varchar(128)             default ''::character varying not null,
    "ProductPrice"                 numeric(14, 2)           default 0                     not null,
    "ProductPromotedPrice"         numeric(14, 2)           default 0                     not null,
    "ProductVAT"                   numeric(6, 6)            default 0                     not null,
    "ProductCostPrice"             numeric(14, 2)           default 0                     not null,
    "ProductType"                  smallint                 default 0                     not null,
    "ProductExpiryDate"            integer                  default 0                     not null,
    "ProductColourID"              smallint                 default 0                     not null,
    "ProductSlots"                 smallint                 default 0                     not null,
    "ProductVoucherUnits"          bigint                   default 0                     not null,
    "ProductVoucherUnitClassID"    smallint                 default 1                     not null,
    "ProductVoucherZCID"           bigint,
    "ProductVoucherPID"            bigint,
    "ProductVoucherNum"            bigint,
    "ProductVoucherExpiryInterval" text                     default ''::text              not null,
    "ProductVoucherExpiryUnits"    integer                  default 0                     not null,
    "ProductVoucherExpiryDate"     date,
    "ProductVoucherVariable"       smallint                 default 0                     not null,
    "ProductVoucherDepositPcnt"    numeric(7, 6),
    "ProductShotsMin"              integer                  default 0                     not null,
    "ProductShotsMax"              integer                  default 0                     not null,
    "ProductTimerMin"              integer                  default 0                     not null,
    "ProductTimerMax"              integer                  default 0                     not null,
    "ProductRequiresDoctor"        smallint                 default 0                     not null,
    "ProductStatsDisabled"         smallint                 default 0                     not null,
    "ProductFWAEnabled"            smallint                 default 0                     not null,
    "ProductSchedAndBuy"           smallint                 default 0                     not null,
    "ProductEvaluation"            smallint                 default 0                     not null,
    "ProductCabinets"              varchar(5000)            default ''::character varying not null,
    "ProductEquipmentID"           bigint                   default 0                     not null,
    "ProductDisabled"              smallint                 default 0                     not null,
    "ProductStamp"                 timestamp,
    "ProductOrder"                 smallint                 default 0                     not null,
    "ProductJoinCitation"          smallint                 default 0                     not null,
    "ProductDynamicPackCreation"   smallint                 default 0                     not null,
    "ProductIsGiftable"            smallint                 default 0                     not null,
    "ProductTCEnabled"             smallint                 default 0                     not null,
    _ingested_at                   timestamp with time zone default now()                 not null,
    _ingestion_batch_id            bigint
        references ingestion_batches
);

alter table x_config_products_det
    owner to admin;

create table __x_config_users_view
(
    "UserID"            bigint                                                 not null
        primary key,
    "UserLogin"         varchar(255),
    "UserName"          varchar(255)             default ''::character varying not null,
    "UserSex"           text                     default 'F'::text             not null,
    "UserDisabled"      smallint                 default 0                     not null,
    _ingested_at        timestamp with time zone default now()                 not null,
    _ingestion_batch_id bigint
        references ingestion_batches
);

alter table __x_config_users_view
    owner to admin;

create table __x_config_workplaces_view
(
    "WorkPlaceID"          bigint                                 not null
        primary key,
    "WorkPlaceUserID"      bigint                   default 0     not null,
    "WorkPlaceClinicID"    integer                  default 0     not null,
    "WorkPlaceUserClassID" smallint                 default 0     not null,
    "WorkPlaceDefault"     smallint                 default 0     not null,
    _ingested_at           timestamp with time zone default now() not null,
    _ingestion_batch_id    bigint
        references ingestion_batches
);

alter table __x_config_workplaces_view
    owner to admin;

create table clients
(
    "ClientID"               bigint                                                 not null
        primary key,
    "ClientClinicID"         integer                  default 0                     not null,
    "ClientNumber"           bigint,
    "ClientName"             varchar(255)             default ''::character varying not null,
    "ClientSurname1"         varchar(64)              default ''::character varying not null,
    "ClientSurname2"         varchar(64)              default ''::character varying not null,
    "ClientSex"              text                     default 'F'::text             not null,
    "ClientBirthDate"        date,
    "ClientCity"             varchar(64)              default ''::character varying not null,
    "ClientProvince"         varchar(255)             default ''::character varying not null,
    "ClientProvinceID"       bigint                   default 0                     not null,
    "ClientCountry"          varchar(255)             default ''::character varying not null,
    "ClientCountryID"        smallint                 default 0                     not null,
    "ClientPostCode"         varchar(32)              default ''::character varying not null,
    "ClientRefererID"        bigint                   default 0                     not null,
    "ClientSource"           varchar(16)              default ''::character varying not null,
    "ClientZones"            varchar(255)             default ''::character varying not null,
    "ClientDate"             date,
    "ClientTime"             time,
    "ClientLock_Enable"      smallint                 default 0                     not null,
    "ClientVIP"              smallint                 default 0                     not null,
    "ClientFWA"              smallint                 default 0                     not null,
    "ClientJobClassID"       smallint                 default 0                     not null,
    "ClientUserID"           bigint                   default '-1'::integer         not null,
    "ClientStamp"            timestamp,
    "ClientLifeCycleStageID" bigint                   default 1                     not null,
    _ingested_at             timestamp with time zone default now()                 not null,
    _ingestion_batch_id      bigint
        references ingestion_batches
);

alter table clients
    owner to admin;

create table vouchers
(
    "VoucherID"                   bigint                                 not null
        primary key,
    "VoucherGID"                  bigint,
    "VoucherTicketID"             bigint                   default 0     not null,
    "VoucherCreditNoteID"         bigint                   default 0     not null,
    "VoucherClientID"             bigint                   default 0     not null,
    "VoucherUserID"               bigint                   default 0     not null,
    "VoucherImportID"             bigint                   default 0     not null,
    "VoucherProductID"            bigint                   default 0     not null,
    "VoucherUnits"                integer                  default 1     not null,
    "VoucherPackID"               bigint                   default 0     not null,
    "VoucherAvailable"            integer                  default 0     not null,
    "VoucherUsed"                 bigint                   default 0     not null,
    "VoucherTransferred"          bigint                   default 0     not null,
    "VoucherCancelled"            bigint                   default 0     not null,
    "VoucherDisabled"             bigint                   default 0     not null,
    "VoucherDate"                 timestamp,
    "VoucherDateCreated"          timestamp,
    "VoucherExpiryDate"           date,
    "VoucherStatTotalAmount"      numeric(14, 2)           default 0     not null,
    "VoucherStatOriginalSessions" integer                  default 0     not null,
    "VoucherMigrated"             smallint                 default 0     not null,
    "VoucherNotRegularized"       smallint                 default 0     not null,
    "VoucherMigratedPrice"        numeric(14, 2)           default 0     not null,
    "VoucherMPG"                  smallint                 default 0     not null,
    "VoucherSignatureID"          bigint                   default 0     not null,
    "VoucherClosed"               smallint                 default 0     not null,
    _ingested_at                  timestamp with time zone default now() not null,
    _ingestion_batch_id           bigint
        references ingestion_batches
);

alter table vouchers
    owner to admin;

create table diary_gen
(
    "DiaryGID"                bigint                                                 not null
        primary key,
    "DiaryGGroupID"           bigint                   default 0                     not null,
    "DiaryGMultID"            bigint                   default 0                     not null,
    "DiaryGClientID"          bigint                   default 0                     not null,
    "DiaryGClinicID"          integer                  default 0                     not null,
    "DiaryGCabinetNum"        smallint                 default 0                     not null,
    "DiaryGUserID"            bigint                   default 0                     not null,
    "DiaryGSUserID"           bigint                   default 0                     not null,
    "DiaryGUserClassID"       smallint                 default 0                     not null,
    "DiaryGPackID"            bigint,
    "DiaryGDate"              date,
    "DiaryGStart"             time,
    "DiaryGEnd"               time,
    "DiaryGColourID"          smallint                 default 0                     not null,
    "DiaryGLaserGID"          bigint                   default 0                     not null,
    "DiaryGLaserDoctor"       smallint                 default 0                     not null,
    "DiaryGLockID"            bigint                   default 0                     not null,
    "DiaryGLocked"            smallint                 default 0                     not null,
    "DiaryGUsesProtocol"      smallint                 default 0                     not null,
    "DiaryGNotUsesInterval"   smallint                 default 0                     not null,
    "DiaryGFWA"               smallint                 default 0                     not null,
    "DiaryGDelayed"           smallint                 default 0                     not null,
    "DiaryGDelayedID"         bigint                   default 0                     not null,
    "DiaryGNotAttended"       smallint                 default 0                     not null,
    "DiaryGNotAttendedReason" text,
    "DiaryGTrackingUID"       bigint                   default 0                     not null,
    "DiaryGTrackingStamp"     timestamp,
    "DiaryGConfirmed"         smallint                 default 0                     not null,
    "DiaryGComments"          text,
    "DiaryGCreated"           timestamp,
    "DiaryGStamp"             timestamp,
    "DiaryGMigrated"          smallint                 default 0                     not null,
    "DiaryGInvisible"         smallint                 default 0                     not null,
    "DiaryGTags"              varchar(255)             default ''::character varying not null,
    "DiaryGCompositionID"     bigint                   default 0                     not null,
    "DiaryGTemporalLock"      smallint                 default 0                     not null,
    "DiaryGTCEnabled"         smallint                 default 0                     not null,
    _ingested_at              timestamp with time zone default now()                 not null,
    _ingestion_batch_id       bigint
        references ingestion_batches
);

alter table diary_gen
    owner to admin;

create table diary_det
(
    "DiaryID"           bigint                                 not null
        primary key,
    "DiaryGID"          bigint                   default 0     not null,
    "DiaryBudgetID"     bigint                   default 0     not null,
    "DiaryZoneID"       bigint                   default 0     not null,
    "DiaryPackID"       bigint                   default 0     not null,
    "DiaryVoucherID"    bigint                   default 0     not null,
    "DiarySessionID"    smallint                 default 0     not null,
    _ingested_at        timestamp with time zone default now() not null,
    _ingestion_batch_id bigint
        references ingestion_batches
);

alter table diary_det
    owner to admin;

create table laser_gen
(
    "LaserGID"            bigint                                 not null
        primary key,
    "LaserGClientID"      bigint                   default 0     not null,
    "LaserGClinicID"      integer                  default 0     not null,
    "LaserGCabinetNum"    bigint                   default 0     not null,
    "LaserGUserID"        bigint                   default 0     not null,
    "LaserGCloseDate"     timestamp,
    "LaserGDate"          date,
    "LaserGStart"         time,
    "LaserGEnd"           time,
    "LaserGClientDelayed" smallint                 default 0     not null,
    "LaserGComments"      text,
    "LaserGTicketGID"     bigint                   default 0     not null,
    "LaserGBudgetGID"     bigint                   default 0     not null,
    "LaserGStamp"         timestamp,
    _ingested_at          timestamp with time zone default now() not null,
    _ingestion_batch_id   bigint
        references ingestion_batches
);

alter table laser_gen
    owner to admin;

create table laser_det
(
    "LaserID"                bigint                                                 not null
        primary key,
    "LaserGID"               bigint                   default 0                     not null,
    "LaserBudgetID"          bigint                   default 0                     not null,
    "LaserZoneID"            bigint                   default 0                     not null,
    "LaserZoneDesc"          varchar(255)             default ''::character varying not null,
    "LaserSessionID"         integer                  default 0                     not null,
    "LaserImportID"          bigint                   default 0                     not null,
    "LaserEquipmentID"       bigint                   default 0                     not null,
    "LaserClinicEquipmentID" bigint                   default 0                     not null,
    "LaserSerialNum"         varchar(255)             default ''::character varying not null,
    "LaserShowClinicHistory" smallint                 default 0                     not null,
    _ingested_at             timestamp with time zone default now()                 not null,
    _ingestion_batch_id      bigint
        references ingestion_batches
);

alter table laser_det
    owner to admin;

create table tickets_gen
(
    "TicketGID"             bigint                                                 not null
        primary key,
    "TicketGInvoiceID"      bigint                   default 0                     not null,
    "TicketGParentSourceID" bigint                   default 0                     not null,
    "TicketGBudgetID"       bigint                   default 0                     not null,
    "TicketGClientID"       bigint                   default 0                     not null,
    "TicketGUserID"         bigint                   default 0                     not null,
    "TicketGSUserID"        bigint                   default 0                     not null,
    "TicketGClinicID"       integer                  default 0                     not null,
    "TicketGNumber"         bigint                   default 0                     not null,
    "TicketGYear"           smallint,
    "TicketGDate"           date,
    "TicketGTime"           time,
    "TicketGDateRegen"      timestamp,
    "TicketGClosed"         smallint                 default 0                     not null,
    "TicketGErased"         smallint                 default 0                     not null,
    "TicketGFWA"            bigint                   default 0                     not null,
    "TicketGCancelled"      bigint                   default 0                     not null,
    "TicketGBaseAmount"     numeric(14, 2)           default 0                     not null,
    "TicketGVATAmount"      numeric(14, 2)           default 0                     not null,
    "TicketGTotalAmount"    numeric(14, 2)           default 0                     not null,
    "TicketGTotalSold"      numeric(14, 2)           default 0                     not null,
    "TicketGPromotionCode"  varchar(32)              default ''::character varying not null,
    "TicketGSimulated"      smallint                 default 0                     not null,
    _ingested_at            timestamp with time zone default now()                 not null,
    _ingestion_batch_id     bigint
        references ingestion_batches
);

alter table tickets_gen
    owner to admin;

create table tickets_det
(
    "TicketID"                    bigint                                 not null
        primary key,
    "TicketProductCID"            bigint,
    "TicketGID"                   bigint                   default 0     not null,
    "TicketBudgetID"              bigint                   default 0     not null,
    "TicketUserID"                bigint                   default 0     not null,
    "TicketLaserGID"              bigint                   default 0     not null,
    "TicketLaserID"               bigint                   default 0     not null,
    "TicketProductID"             bigint                   default 0     not null,
    "TicketUnits"                 integer                  default 1     not null,
    "TicketPrice"                 numeric(14, 4)           default 0     not null,
    "TicketPriceVAT"              numeric(14, 2)           default 0     not null,
    "TicketDiscount"              numeric(16, 15)          default 0     not null,
    "TicketDiscountUnitAmountVAT" numeric(14, 4)           default 0     not null,
    "TicketVAT"                   numeric(6, 6)            default 0     not null,
    "TicketTotalAmount"           numeric(14, 2)           default 0     not null,
    "TicketPayVoucherID"          bigint                   default 0     not null,
    "TicketPaySubscription"       smallint                 default 0     not null,
    "TicketVoucherUnits"          bigint                   default 0     not null,
    "TicketStatSaleAmount"        numeric(14, 2)           default 0     not null,
    "TicketTotalSold"             numeric(14, 2)           default 0     not null,
    "TicketTimeStamp"             timestamp,
    _ingested_at                  timestamp with time zone default now() not null,
    _ingestion_batch_id           bigint
        references ingestion_batches
);

alter table tickets_det
    owner to admin;

