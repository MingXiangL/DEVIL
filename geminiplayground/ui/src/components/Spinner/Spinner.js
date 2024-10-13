import styles from './styles.module.css'

export default function Spinner({className}) {
    return (
        <div className={`${styles.loader} ${className}`}>
            <span></span>
            <span></span>
            <span></span>
        </div>
    )
}