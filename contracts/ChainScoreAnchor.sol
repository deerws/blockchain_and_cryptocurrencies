// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ChainScoreAnchor
 * @notice Anchors credit score hashes on-chain for auditability.
 *
 * The ChainScore model runs off-chain (ML inference is too expensive on-chain).
 * This contract stores a cryptographic commitment to each score so that:
 *   - Lenders can verify a score was issued at a specific time.
 *   - Wallets can prove their historical score without trusting ChainScore servers.
 *   - Score updates are publicly auditable via on-chain events.
 *
 * Score hash = keccak256(abi.encodePacked(wallet, score, validUntil, modelVersion))
 *
 * Deployment target: Sepolia testnet (for MVP demo).
 */
contract ChainScoreAnchor {

    // ── Storage ──────────────────────────────────────────────────────────

    struct ScoreRecord {
        bytes32 scoreHash;      // Commitment to (wallet, score, validUntil, modelVersion)
        uint256 issuedAt;       // Block timestamp when score was anchored
        uint256 validUntil;     // Timestamp after which score should be refreshed
        address issuer;         // Oracle address that submitted the score
        uint16  modelVersion;   // Incrementing model version number
    }

    /// @dev oracle is the only address allowed to anchor scores
    address public oracle;

    /// @dev pending oracle — two-step transfer to avoid accidental lockout
    address public pendingOracle;

    /// @dev wallet → latest score record
    mapping(address => ScoreRecord) public scores;

    /// @dev wallet → historical score count (for audit trail length)
    mapping(address => uint256) public scoreCount;

    // ── Events ───────────────────────────────────────────────────────────

    event ScoreAnchored(
        address indexed wallet,
        bytes32 indexed scoreHash,
        uint256 validUntil,
        uint16  modelVersion,
        address issuer
    );

    event OracleTransferInitiated(address indexed currentOracle, address indexed pendingOracle);
    event OracleTransferred(address indexed oldOracle, address indexed newOracle);

    // ── Errors ───────────────────────────────────────────────────────────

    error NotOracle();
    error NotPendingOracle();
    error ZeroAddress();
    error InvalidValidUntil();

    // ── Constructor ──────────────────────────────────────────────────────

    constructor(address _oracle) {
        if (_oracle == address(0)) revert ZeroAddress();
        oracle = _oracle;
    }

    // ── Modifiers ────────────────────────────────────────────────────────

    modifier onlyOracle() {
        if (msg.sender != oracle) revert NotOracle();
        _;
    }

    // ── Core functions ───────────────────────────────────────────────────

    /**
     * @notice Anchor a new credit score for a wallet.
     * @param wallet        The borrower's Ethereum address.
     * @param scoreHash     keccak256(abi.encodePacked(wallet, score, validUntil, modelVersion))
     * @param validUntil    Unix timestamp after which the score expires (≤ 31 days from now).
     * @param modelVersion  Incrementing version of the ML model that issued this score.
     */
    function anchorScore(
        address wallet,
        bytes32 scoreHash,
        uint256 validUntil,
        uint16  modelVersion
    ) external onlyOracle {
        if (wallet == address(0)) revert ZeroAddress();
        if (validUntil <= block.timestamp) revert InvalidValidUntil();

        scores[wallet] = ScoreRecord({
            scoreHash:    scoreHash,
            issuedAt:     block.timestamp,
            validUntil:   validUntil,
            issuer:       msg.sender,
            modelVersion: modelVersion
        });

        unchecked { scoreCount[wallet]++; }

        emit ScoreAnchored(wallet, scoreHash, validUntil, modelVersion, msg.sender);
    }

    /**
     * @notice Verify that a claimed score matches the on-chain commitment.
     * @param wallet        Wallet to verify.
     * @param score         The numeric score (0–1000) claimed by the holder.
     * @param validUntil    The validity timestamp embedded in the score.
     * @param modelVersion  The model version embedded in the score.
     * @return valid        True if the hash matches and the score has not expired.
     */
    function verifyScore(
        address wallet,
        uint256 score,
        uint256 validUntil,
        uint16  modelVersion
    ) external view returns (bool valid) {
        ScoreRecord memory record = scores[wallet];
        if (record.issuedAt == 0) return false;
        if (block.timestamp > record.validUntil) return false;

        bytes32 expected = keccak256(
            abi.encodePacked(wallet, score, validUntil, modelVersion)
        );
        return record.scoreHash == expected;
    }

    /**
     * @notice Return the latest score record for a wallet (without verifying values).
     */
    function getScore(address wallet)
        external view
        returns (bytes32 scoreHash, uint256 issuedAt, uint256 validUntil, uint16 modelVersion)
    {
        ScoreRecord memory r = scores[wallet];
        return (r.scoreHash, r.issuedAt, r.validUntil, r.modelVersion);
    }

    // ── Oracle management ────────────────────────────────────────────────

    /**
     * @notice Initiate an oracle address transfer (two-step for safety).
     */
    function transferOracle(address newOracle) external onlyOracle {
        if (newOracle == address(0)) revert ZeroAddress();
        pendingOracle = newOracle;
        emit OracleTransferInitiated(oracle, newOracle);
    }

    /**
     * @notice Complete the oracle transfer — must be called by the pending address.
     */
    function acceptOracle() external {
        if (msg.sender != pendingOracle) revert NotPendingOracle();
        emit OracleTransferred(oracle, pendingOracle);
        oracle = pendingOracle;
        pendingOracle = address(0);
    }
}
